"""
Training time-dependent FNO
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset


from utils import (
   train_model,
   get_experiment_name,
   save_config,
)
from visualization import plot_training_history
from fno import FNO1d
from dataset import All2All


def main(data_mode="onetoone"):
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
        "nfun": 2,
        "time_dependent": True,
        "model_type": "custom",
        "device": device
    }
    
    # Training configuration
    training_config = {
        'n_train': 64,
        'batch_size': 5,
        'learning_rate': 0.001,
        'epochs': 100,
        'step_size': 100,
        'gamma': 0.1,
        'patience': 10,
        'freq_print': 1,
        'device': device
    }

    # Combine configurations for experiment naming
    naming_config = {**model_config, 'learning_rate': training_config['learning_rate']}

    # Create experiment directory
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/custom_fno/time") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save full configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    #================================
    # with custom All2All dataset loading
    #================================
    # Dataset selection
    batch_size = training_config['batch_size']
    if data_mode == "onetoone":
        DataClass = OneToOne
    elif data_mode == "all2all":
        DataClass = All2All
    else:
        raise ValueError("data_mode must be 'onetoone' or 'all2all'")

    print(f"Training strategy specified as {data_mode}")
    training_set = DataLoader(DataClass("training"), batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(DataClass("validation"), batch_size=batch_size, shuffle=False)    

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
    main("all2all")  # or main("all2all")