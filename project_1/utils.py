"""
Utility functions used for training and evaluating the FNO models
"""

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from fno import FNO1d
from torch.utils.data import DataLoader
from dataset import OneToOne, All2All

def print_bold(text: str) -> None:
    """Print a bold header text."""
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{BOLD}{text}{RESET}")

def get_experiment_name(config):
    """Create a unique experiment name based on key parameters and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fno_m{config['modes']}_w{config['width']}_d{config['depth']}_lr{config['learning_rate']}_{timestamp}"

def save_config(config, save_dir):
    """Save configuration to a JSON file"""
    config_copy = {
        'model_config': {k: str(v) if isinstance(v, torch.device) else v 
                        for k, v in config['model_config'].items()},
        'training_config': {k: str(v) if isinstance(v, torch.device) else v 
                          for k, v in config['training_config'].items()}
    }
    
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config_copy, f, indent=4)

def load_model(checkpoint_dir: str) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'training_config.json', 'r') as f:
        config_dict = json.load(f)
    
    model_config = config_dict['model_config']
    model_args = {k: v for k, v in model_config.items()}
    model = FNO1d(**model_args)
    
    model.load_state_dict(torch.load(checkpoint_dir / 'model.pth', weights_only=True))    
    model = model.to(device)
    model.eval()
    return model

def train_model(model, training_set, testing_set, config, checkpoint_dir):
    """
    Model-agnostic training function.
    """    
    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['learning_rate'],
                                weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=config['step_size'], 
                                              gamma=config['gamma'])
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    criterion = torch.nn.MSELoss()
        
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    for epoch in range(config['epochs']):
        model.train()
        train_mse = 0.0
        
        for input_batch, output_batch in training_set:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            optimizer.zero_grad()
            output_pred_batch = model(input_batch)
            loss = criterion(output_pred_batch, output_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_mse += loss.item()
        
        train_mse /= len(training_set)
        scheduler.step()
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for input_batch, output_batch in testing_set:
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output_pred_batch = model(input_batch)
                loss = criterion(output_pred_batch, output_batch)
                val_loss += loss.item()
            val_loss /= len(testing_set)
        
        training_history['train_loss'].append(train_mse)
        training_history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            epochs_without_improvement = 0
            
            torch.save(model.state_dict(), checkpoint_dir / 'model.pth')
            
            try:
                with open(checkpoint_dir / 'training_config.json', 'r') as f:
                    full_config = json.load(f)
            except FileNotFoundError:
                full_config = {'training_config': config}
            
            full_config['training_history'] = training_history
            
            with open(checkpoint_dir / 'training_config.json', 'w') as f:
                json.dump(full_config, f, indent=4)
                
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % config['freq_print'] == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_mse:.6f}, Validation Loss: {val_loss:.6f}")
    
    return model, training_history

def evaluate_direct(model, 
                  data_path: str,
                  batch_size=5, 
                  start_idx=0, 
                  end_idx=4,
                  dt=0.25,
                  strategy="onetoone",
                  time_pairs=None,
                  device=None):
    """Using direct inference to evaluate the performance of a given model on a testing dataset."""
    
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()

    if strategy == "onetoone":
        strategy = OneToOne("testing", data_path=data_path, start_idx=start_idx, end_idx=end_idx, device=device)
    elif strategy == "all2all" and not time_pairs:
        strategy = All2All("testing", data_path=data_path, device=device, dt=dt)
    elif strategy == "all2all" and time_pairs:
        strategy = All2All("testing", data_path=data_path, time_pairs =time_pairs, device=device)
    else:
        raise ValueError("Invalid strategy. Please choose either 'onetoone' or 'all2all'.")

    test_loader = DataLoader(strategy, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_errors = torch.tensor([], device=device)
    u0, uT = [], []
    
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            predictions = model(batch_input)
            predictions = predictions.squeeze(-1)
            targets = batch_target.squeeze(-1)
            
            u0.append(batch_input[:, 0].cpu())
            uT.append(targets.cpu())
            
            # Relative L2 norm
            individual_abs_errors = torch.norm(predictions - targets, p=2, dim=1) / torch.norm(targets, p=2, dim=1)

            # Record current prediction and error
            all_predictions.append(predictions.cpu())
            all_errors = torch.cat([all_errors, individual_abs_errors])

        # Concatenate all predictions in order
        predictions = torch.cat(all_predictions, dim=0)

        # Calculate average relative L2 error
        average_error = all_errors.mean().item() * 100  # (in %)
        
        results = {
            'predictions': predictions,
            'error': average_error,
            'individual_errors': (all_errors * 100).cpu().tolist()  # (in %)
        }
    
    u0 = torch.cat(u0, dim=0)
    uT = torch.cat(uT, dim=0)
    
    return results, (u0, uT)

def evaluate_autoregressive(model, 
                            data_path, 
                            timesteps, 
                            batch_size=5,
                            base_dt = 0.25,
                            start_idx = 0,
                            end_idx = 4,
                            device="cuda"):
    """Using autoregressive inference to evaluate the performance of a given model on a testing dataset."""
    model.eval()
    strategy = All2All("testing", 
                       data_path=data_path,
                       time_pairs=[(start_idx, end_idx)],
                       dt=base_dt,
                       device=device)

    test_loader = DataLoader(strategy, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_errors = torch.tensor([], device=device)
    u0, uT = [], []
    
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            u_start = batch_input[..., 0]
            v_start = batch_input[..., 1]  # This is already central-differenced from All2All
            x_grid = batch_input[..., 2]
            
            u_current = u_start
            v_current = v_start
            
            for step_size in timesteps:
                dt = torch.full_like(u_start, base_dt * step_size, device=device)
                
                current_input = torch.stack(
                    (u_current, v_current, x_grid, dt), dim=-1
                )
                
                u_next = model(current_input).squeeze(-1)
                # Forward difference for velocity (matching All2All's treatment of initial velocity)
                v_next = (u_next - u_current) / dt
                u_current = u_next
                v_current = v_next
            
            predictions = u_current
            targets = batch_target.squeeze(-1)
            
            u0.append(batch_input[..., 0].cpu())
            uT.append(targets.cpu())
            
            individual_abs_errors = torch.norm(predictions - targets, p=2, dim=1) / torch.norm(targets, p=2, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_errors = torch.cat([all_errors, individual_abs_errors])

    predictions = torch.cat(all_predictions, dim=0)
    average_error = all_errors.mean().item() * 100
    
    results = {
        'predictions': predictions,
        'error': average_error,
        'individual_errors': (all_errors * 100).cpu().tolist()
    }
    
    u0 = torch.cat(u0, dim=0)
    uT = torch.cat(uT, dim=0)
    
    return results, (u0, uT)