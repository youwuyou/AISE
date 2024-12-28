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
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from time_training import (
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
        
        # Simple linear transformations for scale and bias
        self.inp2scale = nn.Linear(1, channels, bias=True)
        self.inp2bias = nn.Linear(1, channels, bias=True)
        
        # Initialize to identity mapping (very important!)
        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(1)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)
        
        # Use BatchNorm for stability
        self.norm = nn.BatchNorm1d(channels) if use_bn else nn.Identity()

    def forward(self, x, time):
        # x shape: [batch, channels, sequence]
        x = self.norm(x)
        
        time = time.unsqueeze(1).float()
        scale = self.inp2scale(time).unsqueeze(-1)
        bias = self.inp2bias(time).unsqueeze(-1)
        
        return x * scale + bias


class FNO1d(nn.Module):
    def __init__(self, depth, modes, width, device="cuda", nfun=1, padding_frac=1/4):
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.depth = depth
        self.padding_frac = padding_frac
        
        # Input lifting layer
        self.fc0 = nn.Linear(nfun + 1, self.width)
        
        # Fourier and conv layers
        self.spectral_list = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)
        ])
        
        self.w_list = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(self.depth)
        ])
                
        # Time-conditional normalization
        self.film_list = nn.ModuleList([
            FILM(self.width, use_bn=True)  # Enable BatchNorm for stability
            for _ in range(self.depth)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, nfun)
                
        self.activation = nn.GELU()
        self.device = device
        self.to(device)

    def forward(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        
        # Lift
        x = x.transpose(1, 2)
        x = self.fc0(x)
        x = x.transpose(1, 2)
        
        # Add padding if specified
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        if x_padding > 0:
            x = F.pad(x, [0, x_padding])
        
        # Main network backbone
        for i in range(self.depth):
            # Fourier layer
            x1 = self.spectral_list[i](x)
            
            # Residual layer
            x2 = self.w_list[i](x)
            
            # Simple residual connection
            x = x1 + x2
            
            # Time-conditional normalization
            x = self.film_list[i](x, t)
            
            if i < self.depth - 1:
                x = self.activation(x)
        
        # Remove padding
        if x_padding > 0:
            x = x[..., :-x_padding]
        
        # Project back to physical space
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x.transpose(1, 2)

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

    model_config = {
        "depth": 4,
        "modes": 30,
        "width": 32,
    }

    training_config = {
        'batch_size': 5,
        'learning_rate': 0.001,
        'epochs': 400,
        'step_size': 100,
        'gamma': 0.5,
        'weight_decay': 1e-6,
        'optimizer': 'AdamW',
        'patience': 40,
        'freq_print': 1,
        'training_mode': 'all2all',
        'grad_clip': 0.5,
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

    # Setup data and model
    model = FNO1d(**model_config)
    
    # Create dataset with debugging
    dataset = PDEDataset(
        data_path="data/train_sol.npy",
        training_samples=64,
        training_mode="all2all",
        total_time=1.0  # Make sure this matches your PDE setup
    )
    
    # Debug a sample batch
    dataset.debug_batch(batch_size=5)
    
    # Get samplers for training and validation
    train_sampler, val_sampler = dataset.get_train_val_samplers()

    # Create DataLoaders with reduced workers
    train_loader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        sampler=train_sampler,
        num_workers=1  # Reduced to avoid warnings
    )

    val_loader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        sampler=val_sampler,
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

    # Use time-specific training function
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