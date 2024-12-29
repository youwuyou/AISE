"""
Uses custom FNO implementation for training (independent of time)
"""

import torch
import numpy as np
from pathlib import Path
from training import train_model, get_experiment_name, save_config
from fno import FNO1d
from torch.utils.data import DataLoader, TensorDataset, Dataset

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Model configuration
    # checkpoints/custom_fno/fno_m30_w64_d4_lr0.001_20241227_190139
    # 11.99% resolution 32; 4.92% resolution 64
    model_config = {
        "depth": 4,
        "modes": 30,
        "width": 64,
        "model_type": "custom",
        "device": device
    }

    # Training configuration
    training_config = {
        'n_train': 64,
        'batch_size': 5,
        'learning_rate': 0.001,
        'epochs': 400,
        'step_size': 100,
        'gamma': 0.1,
        'patience': 40,
        'freq_print': 1,
        'device': device
    }

    # Combine configurations for experiment naming
    naming_config = {**model_config, 'learning_rate': training_config['learning_rate']}

    # Create experiment directory
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/custom_fno") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save full configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    # Prepare data
    x_grid = torch.linspace(0, 1, 64).float()
    if device:
        x_grid = x_grid.to(device)

    def prepare_input(u0):
        batch_size = u0.shape[0]
        x_grid_expanded = x_grid.expand(batch_size, -1)
        return torch.stack((u0, x_grid_expanded), dim=-1)

    data_path = "data/train_sol.npy"
    n_train   = training_config['n_train']
    batch_size = training_config['batch_size']
    data = torch.from_numpy(np.load(data_path)).type(torch.float32)
    if device:
        data = data.to(device)
        
    u_0_all = data[:, 0, :]   # All initial conditions
    u_T_all = data[:, -1, :]  # All output data
    input_function_train = prepare_input(u_0_all[:n_train, :])
    input_function_test = prepare_input(u_0_all[n_train:, :])
    output_function_train = u_T_all[:n_train, :].unsqueeze(-1)
    output_function_test = u_T_all[n_train:, :].unsqueeze(-1)

    training_set = DataLoader(TensorDataset(input_function_train, output_function_train), 
                            batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), 
                            batch_size=batch_size, shuffle=False)

    # Initialize model with device
    model = FNO1d(**{k: v for k, v in model_config.items() if k != 'model_type'})

    # Train model
    trained_model, training_history = train_model(
        model=model,
        training_set=training_set,
        testing_set=testing_set,
        config=training_config,
        checkpoint_dir=checkpoint_dir
    )

    print(f"Training completed. Model saved in {checkpoint_dir}")
    print(f"Best validation loss: {training_history['best_val_loss']:.6f} at epoch {training_history['best_epoch']}")

if __name__ == "__main__":
   main()