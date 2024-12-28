"""
Code reference and inspiration taken from:

https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/CNO2d_time_dependent_%26_foundation_model/CNO_timeModule_CIN.py
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

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
from custom_fno_1d import SpectralConv1d

class FILM(nn.Module):
    def __init__(self, channels, use_bn=True):
        super(FILM, self).__init__()
        self.channels = channels
        
        # Enhanced linear transformations with intermediate layer
        self.inp2scale = nn.Sequential(
            nn.Linear(1, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.inp2bias = nn.Sequential(
            nn.Linear(1, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        
        # Initialize to identity mapping
        for module in self.inp2scale.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        last_layer.weight.data.fill_(0)
        last_layer.bias.data.fill_(1)
        
        for module in self.inp2bias.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        last_layer.weight.data.fill_(0)
        last_layer.bias.data.fill_(0)
        
        # Use both BatchNorm and LayerNorm for better stability
        self.bn = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        self.ln = nn.LayerNorm([channels])

    def forward(self, x, Δt):
        # x shape: [batch, channels, sequence]
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # [batch, sequence, channels]
        x = self.ln(x)
        x = x.permute(0, 2, 1)  # [batch, channels, sequence]
        
        Δt = Δt.unsqueeze(1).float()
        scale = self.inp2scale(Δt).unsqueeze(-1)
        bias = self.inp2bias(Δt).unsqueeze(-1)
        
        return x * scale + bias


class FNO1d(nn.Module):
    def __init__(self, depth, modes, width, device="cuda", nfun=1, padding_frac=1/4):
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.depth = depth
        self.padding_frac = padding_frac
        
        # Enhanced input lifting layer
        self.fc0 = nn.Sequential(
            nn.Linear(nfun + 1, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width)
        )
        
        # Fourier and conv layers
        self.spectral_list = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)
        ])
        
        self.w_list = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(self.depth)
        ])
                
        # Time-conditional normalization
        self.film_list = nn.ModuleList([
            FILM(self.width, use_bn=True) for _ in range(self.depth)
        ])
        
        # Enhanced projection layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.width, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.fc2 = nn.Linear(128, nfun)
                
        self.activation = nn.GELU()

        self.device = device
        self.to(device)

    def forward(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        
        # Lift the input
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Add padding if specified
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        if x_padding > 0:
            x = F.pad(x, [0, x_padding], mode='reflect')
        
        # Main network backbone with enhanced residual connections
        for i in range(self.depth):
            x_input = x
            
            # Fourier layer
            x1 = self.spectral_list[i](x)
            
            # Residual layer
            x2 = self.w_list[i](x)
            
            # Enhanced residual connection with scaling
            x = x1 + x2 + 0.1 * x_input
            
            # Time-conditional normalization
            x = self.film_list[i](x, t)
            
            if i < self.depth - 1:
                x = self.activation(x)
        
        # Remove padding
        if x_padding > 0:
            x = x[..., :-x_padding]
        
        # Project back to physical space
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        
        return x
        
    def print_size(self):
        nparams = sum(p.numel() for p in self.parameters())
        print(f'Total number of model parameters: {nparams}')
        return nparams

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
    # Task 4: Testing on Vanilla Training:
    # Direct inference - Average relative L2 error at t_1 = 0.25: 26.2301%
    # Direct inference - Average relative L2 error at t_2 = 0.50: 33.3457%
    # Direct inference - Average relative L2 error at t_3 = 0.75: 37.0793%
    # Direct inference - Average relative L2 error at t_4 = 1.00: 39.7440%
    # Autoregressive - Average relative L2 error at t_1 = 0.50: 33.3457%
    # Autoregressive - Average relative L2 error at t_2 = 1.00: 129.6974%

    # TODO: Compare error to the one from Task 1.

    # Bonus Task: Evaluate Vanilla Training on Different Timesteps:

    # Direct Inference at multiple timesteps
    # Direct inference - Average relative L2 error at t_1 = 0.25: 26.2301%
    # Direct inference - Average relative L2 error at t_2 = 0.50: 33.3457%
    # Direct inference - Average relative L2 error at t_3 = 0.75: 37.0793%
    # Direct inference - Average relative L2 error at t_4 = 1.00: 39.7440%

    # Autoregressive Inference
    # Autoregressive - Average relative L2 error at t_1 = 0.25: 26.2301%
    # Autoregressive - Average relative L2 error at t_2 = 0.50: 70.2454%
    # Autoregressive - Average relative L2 error at t_3 = 0.75: 127.4491%
    # Autoregressive - Average relative L2 error at t_4 = 1.00: 207.6183%

    # Plotting Average Relative L2 Error Across Time

    # Testing OOD Performance:

    # Direct Inference on OOD data:
    # Direct inference - Average relative L2 error at t_1 = 1.00: 51.1273%

    # Autoregressive Inference on OOD data:
    # Autoregressive - Average relative L2 error at t_1 = 0.25: 41.8220%
    # Autoregressive - Average relative L2 error at t_2 = 0.50: 40.6845%
    # Autoregressive - Average relative L2 error at t_3 = 0.75: 46.0933%
    # Autoregressive - Average relative L2 error at t_4 = 1.00: 174.2444%


    model_config = {
        "depth": 4,
        "modes": 30,
        "width": 64,
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
        'training_mode': training_config['training_mode']
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
    
    # Create dataset
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