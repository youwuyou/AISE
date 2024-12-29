"""
Utility functions used for training the FNO models used for both custom and library-based implementation
"""

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

def get_experiment_name(config):
    """Create a unique experiment name based on key parameters and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # For library FNO
    if 'n_modes' in config:
        return f"fno_m{config['n_modes'][0]}_w{config['hidden_channels']}_lr{config['learning_rate']}_{timestamp}"
    # For custom FNO
    else:
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
            
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            
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