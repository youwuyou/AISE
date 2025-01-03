"""
Training time-dependent FNO
- All2All class is used for loading training set
    - supports all2all time pairs loading O(k²)
    - supports one-to-all (vanilla) time pairs loading O(k)
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from fno import FNO1d
from dataset import All2All

from utils import (
   train_model,
   get_experiment_name,
   save_config,
)
from visualization import plot_training_history


def main(data_mode="all2all"):
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
        "depth": 2,
        "modes": 30,
        "width": 32,
        "nfun": 2,
        "time_dependent": True,
        "device": device
    }
    
    # Training configuration
    training_config = {
        "data_mode": data_mode,
        'n_train': 64,
        'batch_size': 5,
        'learning_rate': 0.001,
        'epochs': 800,
        'step_size': 300,
        'gamma': 0.1,
        'patience': 80,
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

    #==================================================
    # with All2All or One-to-All (Vanilla) data loading
    #==================================================
    # Dataset selection
    batch_size = training_config['batch_size']

    print(f"Training strategy specified as {data_mode}")
    training_set = DataLoader(All2All("training", data_mode=data_mode), batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(All2All("validation", data_mode=data_mode), batch_size=batch_size, shuffle=False)    

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['all2all', 'onetoall'], default='all2all')
    args = parser.parse_args()
    
    main(args.mode)