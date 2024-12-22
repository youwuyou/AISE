import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

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
    u_0_all = data[:, 0, :]  # All initial conditions
    u_4_all = data[:, 4, :]  # All output data
    
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
        output_function_train = u_4_all[:n_train, :].unsqueeze(1).unsqueeze(-1)
        output_function_test = u_4_all[n_train:, :].unsqueeze(1).unsqueeze(-1)
    else:
        output_function_train = u_4_all[:n_train, :].unsqueeze(-1)
        output_function_test = u_4_all[n_train:, :].unsqueeze(-1)
    
    # Create DataLoaders
    training_set = DataLoader(TensorDataset(input_function_train, output_function_train), 
                            batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), 
                           batch_size=batch_size, shuffle=False)
    
    return training_set, testing_set, (input_function_test, output_function_test)

def train_model(model, training_set, testing_set, config, checkpoint_dir, experiment_name=None):
    """
    Unified training function that works with both custom and library FNO models.
    
    Args:
        model: The model to train (either custom FNO or library FNO)
        training_set: DataLoader for training data
        testing_set: DataLoader for testing data
        config (dict): Training configuration parameters
        checkpoint_dir (str): Directory to save checkpoints
        experiment_name (str, optional): Name for the experiment, used in checkpoint path
    """
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=config['step_size'], 
                                              gamma=config['gamma'])
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    criterion = torch.nn.MSELoss()
    
    # Create checkpoint directory with experiment name if provided
    checkpoint_dir = Path(checkpoint_dir)
    if experiment_name:
        checkpoint_dir = checkpoint_dir / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_mse,
                'config': config,
                'training_history': training_history
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % config['freq_print'] == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_mse:.6f}, Validation Loss: {val_loss:.6f}")
    
    return model, training_history
