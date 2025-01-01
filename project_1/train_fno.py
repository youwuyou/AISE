"""
Uses custom FNO implementation for training (independent of time)
"""

import torch
import numpy as np
from pathlib import Path
from fno import FNO1d
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import (
train_model, 
get_experiment_name,
save_config
)

from dataset import OneToOne

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
    model_config = {
        "depth": 4,
        "modes": 30,
        "width": 64,
        "nfun": 2,
        "time_dependent": False,
        "device": device
    }

    # Training configuration
    data_mode = "onetoone"
    training_config = {
        "data_mode": data_mode,
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
    checkpoint_dir = Path(f"checkpoints/{data_mode}") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save full configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    #================================
    # with OneToOne dataset loading
    #================================
    batch_size = training_config['batch_size']
    training_set = DataLoader(OneToOne("training"), batch_size=batch_size, shuffle=True)
    testing_set  = DataLoader(OneToOne("validation"), batch_size=batch_size, shuffle=False)

    # Initialize model with device
    model = FNO1d(**{k: v for k, v in model_config.items()})

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