# training.py
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
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)


def prepare_data(data_path, n_train, batch_size, use_library=False):
    """
    Unified data preparation function that can handle both custom and library FNO formats.
    
    Args:
        data_path (str): Path to the data file
        n_train (int): Number of training samples
        batch_size (int): Batch size
        use_library (bool): If True, prepares data in library FNO format
    """
    data = torch.from_numpy(np.load(data_path)).type(torch.float32)
    u_0_all = data[:, 0, :]   # All initial conditions

    # FIXME: it was 4 before!
    u_T_all = data[:, -1, :]  # All output data
    
    # Create spatial grid
    x_grid = torch.linspace(0, 1, 64).float()
    
    def prepare_input(u0):
        batch_size = u0.shape[0]
        x_grid_expanded = x_grid.expand(batch_size, -1)
        if use_library:
            # Library format: [batch_size, channels, spatial_dim, 1]
            input_data = torch.stack((u0, x_grid_expanded), dim=1)
            return input_data.unsqueeze(-1)
        else:
            # Custom format: [batch_size, spatial_dim, 2]
            return torch.stack((u0, x_grid_expanded), dim=-1)
    
    # Prepare training and test inputs
    input_function_train = prepare_input(u_0_all[:n_train, :])
    input_function_test = prepare_input(u_0_all[n_train:, :])
    
    # Prepare outputs based on format
    if use_library:
        output_function_train = u_T_all[:n_train, :].unsqueeze(1).unsqueeze(-1)
        output_function_test = u_T_all[n_train:, :].unsqueeze(1).unsqueeze(-1)
    else:
        output_function_train = u_T_all[:n_train, :].unsqueeze(-1)
        output_function_test = u_T_all[n_train:, :].unsqueeze(-1)
    
    # Create DataLoaders
    training_set = DataLoader(TensorDataset(input_function_train, output_function_train), 
                            batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), 
                           batch_size=batch_size, shuffle=False)
    
    return training_set, testing_set, (input_function_test, output_function_test)

def train_model(model, training_set, testing_set, config, checkpoint_dir):
    """
    Model-agnostic training function.
    
    Args:
        model: The model to train
        training_set: DataLoader for training data
        testing_set: DataLoader for testing data
        config: Training configuration parameters
        checkpoint_dir (str/Path): Directory to save checkpoints
    """    
    # Create checkpoint directory if it doesn't exist
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
        # Training phase
        model.train()
        train_mse = 0.0
        
        for input_batch, output_batch in training_set:
            optimizer.zero_grad()
            output_pred_batch = model(input_batch)
            loss = criterion(output_pred_batch, output_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_mse += loss.item()
        
        train_mse /= len(training_set)
        scheduler.step()
        
        # Validation phase
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for input_batch, output_batch in testing_set:
                output_pred_batch = model(input_batch)
                loss = criterion(output_pred_batch, output_batch)
                val_loss += loss.item()
            val_loss /= len(testing_set)
        
        # Update training history
        training_history['train_loss'].append(train_mse)
        training_history['val_loss'].append(val_loss)
        
        # Save best model and update history
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            epochs_without_improvement = 0
            
            # Save model weights
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            
            # Save training history to training_config.json
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