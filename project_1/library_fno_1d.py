"""
Uses library-based FNO implementation

https://github.com/neuraloperator/neuraloperator/blob/31b77fde81e705b2e3ed7d8f1f5de767f571b125/neuralop/models/fno.py#L69 
"""

import torch
import numpy as np
from neuralop.models import FNO
from pathlib import Path
import os

from training import train_model, get_experiment_name, save_config, prepare_data


def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Model configuration
    # checkpoints/library_fno/fno_m40_w64_lr0.001_20241226_163648
    # 9.77% resolution 32; 7.19% resolution 64
    model_config = {
        'n_layers': 4,
        'hidden_channels': 64,
        'n_modes': (30, 1),
        'in_channels': 2,
        'out_channels': 1,
        'model_type': 'library'
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
        'freq_print': 1
    }
    
    # Combine configurations for experiment naming
    naming_config = {**model_config, 'learning_rate': training_config['learning_rate']}
    
    # Create experiment directory
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/library_fno") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)
    
    # Prepare data (use library=True for library FNO)
    training_set, testing_set, test_data = prepare_data(
        "data/train_sol.npy",
        training_config['n_train'],
        training_config['batch_size'],
        use_library=True
    )
    
    # Initialize model
    model = FNO(**{k: v for k, v in model_config.items() if k != 'model_type'})
    
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



