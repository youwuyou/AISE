"""
Training time-dependent FNO
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset

from time_training import (
   TrajectorySubset,
   PDEDataset,
   train_model
)
from training import (
   get_experiment_name,
   save_config,
)
from visualization import plot_training_history
from fno import FNO1d


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

    training_mode = 'vanilla'
    model_config = {
        "depth": 4,
        "modes": 30,
        "width": 64,
        "time_dependent": True
    }

    training_config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 400,
        'step_size': 100,
        'gamma': 0.1,
        'weight_decay': 1e-4,
        'optimizer': 'AdamW',
        'patience': 40,
        'grad_clip': 1.0,
        'training_mode': training_mode,
        'freq_print': 1,
        'device': device
    }

    # Add experiment naming configuration
    naming_config = {
        **model_config,
        'learning_rate': training_config['learning_rate'],
        'training_mode': training_config['training_mode'],
        'time_dependent': model_config['time_dependent']  # Add to experiment name
    }

    # Create experiment-specific directory
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/custom_fno/time") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save complete configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    # Setup model
    model = FNO1d(**model_config)
    
    # Create dataset (your existing PDEDataset setup remains the same)
    dataset = PDEDataset(
        data_path="data/train_sol.npy",
        training_samples=64,
        training_mode=training_mode,
        total_time=1.0
    )

    # Prepare train/validation splits at trajectory level
    train_trajectories, val_trajectories = dataset.prepare_splits(random_seed=0)
    
    # Report dataset statistics
    dataset.report_statistics()
    
    # Create train and validation subsets
    train_dataset = TrajectorySubset(dataset, train_trajectories)
    val_dataset = TrajectorySubset(dataset, val_trajectories)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=1
    )
    # Debug first batch from training loader
    print("\nDebugging first training batch:")
    first_batch = next(iter(train_loader))
    time_batch, input_batch, output_batch = first_batch
    print(f"Time batch shape: {time_batch.shape}")
    print(f"Time differences: {time_batch}")
    print(f"Input batch shape: {input_batch.shape}")
    print(f"Output batch shape: {output_batch.shape}")

    # Train the model
    trained_model, history = train_model(
        model=model,
        training_set=train_loader,
        validation_set=val_loader,
        config=training_config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    print(f"Training completed. Best validation loss: {history['best_val_loss']:.6f} "
          f"at epoch {history['best_epoch']}")

if __name__ == "__main__":
    main()