import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from neuralop.models import FNO
from pathlib import Path
import os

from training import train_model, get_experiment_name, save_config, prepare_data

"""
Analysis of the configurations:

Learning Rate Impact:


Best performance: 7.3% error with lr=0.001
Clear pattern of degradation with higher learning rates:

lr=0.0001: 9.77% error (too slow)
lr=0.001: 7.3% error (optimal)
lr=0.005: 15.93% error (too fast)
lr=0.01: 21.78% error (much too fast)




Other Parameter Patterns:


Width: Both 64 and 128 can work well
Modes: Varies between 10-32, with 25 modes showing good performance
Batch size: 5 appears more consistently than 2
Step size: 100 appears more frequently in better configurations
Gamma: 0.1 is used in the better performing configurations

Key Findings:

Learning rate is the most sensitive parameter, with 0.001 being optimal
Higher learning rates (>0.001) lead to significant degradation in performance
The combination of modes=10, width=128, batch_size=5 with the optimal learning rate achieves 7.3% error
The configuration using modes=25, width=64 also shows promise with similar other parameters

The visualization shows two key aspects:

The clear relationship between learning rate and error rate
A comparison of different configuration parameters across implementations
"""

def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Model configuration
    model_config = {
        'n_modes': (32, 1),
        'hidden_channels': 64,
        'in_channels': 2,
        'out_channels': 1,
        'spatial_dim': 2,
        'model_type': 'library'
    }
    
    # Training configuration
    training_config = {
        'n_train': 64,
        'batch_size': 5,
        'learning_rate': 0.001,
        'epochs': 500,
        'step_size': 100,
        'gamma': 0.1,
        'patience': 50,
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



