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

# Make sure these imports point to your own local files
from time_training import (
    PDEDataset, 
    train_model
)
from training import (
    get_experiment_name,
    save_config
)
from visualization import plot_training_history



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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        # Initialize output array
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", 
                                                x_ft[:, :, :self.modes], 
                                                self.weights)
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, depth, modes, width, device="cuda", nfun=1, padding_frac=1/4):
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.depth = depth
        self.padding_frac = padding_frac
        
        # Input lifting layer
        self.fc0 = nn.Linear(2, self.width)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.zeros_(self.fc0.bias)
        
        # Fourier and conv layers
        self.spectral_list = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) 
            for _ in range(self.depth)
        ])
        
        self.w_list = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) 
            for _ in range(self.depth)
        ])
        
        # Initialize conv layers properly
        for conv in self.w_list:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        # Time-conditional normalization
        self.film_list = nn.ModuleList([
            FILM(self.width, use_bn=True)  # Enable BatchNorm for stability
            for _ in range(self.depth)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, nfun)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
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
        # x = self.activation(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x.transpose(1, 2)

    def print_size(self):
        nparams = sum(p.numel() for p in self.parameters())
        print(f'Total number of model parameters: {nparams}')
        return nparams


def main():
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print("GPU device:", torch.cuda.get_device_name(0))

    torch.manual_seed(0)
    np.random.seed(0)

    # Task 4: Testing on All2All Training:
    # Average relative L2 error at t=1.00: 50.1759%

    # TODO: Compare error to the one from Task 1.

    # Bonus Task: Evaluate All2All Training on Different Timesteps:

    # Direct Inference at multiple timesteps
    # Average relative L2 error at t=0.25: 29.2405%
    # Average relative L2 error at t=0.50: 63.3613%
    # Average relative L2 error at t=0.75: 61.3682%
    # Average relative L2 error at t=1.00: 50.1759%

    # Auto regressive at multiple timesteps
    # Average relative L2 error at t=0.25: 29.2405%
    # Average relative L2 error at t=0.50: 50.9501%
    # Average relative L2 error at t=0.75: 64.9494%
    # Average relative L2 error at t=1.00: 82.7478%

    # Testing OOD at t = 1.0
    # Average relative L2 error on OOD data: 44.0449%

    # Visualize with heapmap for direct inference
    # Average relative L2 error at t=0.25: 29.2405%
    # Average relative L2 error at t=0.50: 63.3613%
    # Average relative L2 error at t=0.75: 61.3682%
    # Average relative L2 error at t=1.00: 50.1759%
    # model_config = {
    #     "depth": 4,           # Increase depth for better long-term dependencies
    #     "modes": 96,          # More modes to capture higher frequency components
    #     "width": 64,         # Wider network for more capacity
    #     "device": device,
    # }
    model_config = {
        "depth": 3,           # Reduce depth to prevent overfitting
        "modes": 30,          # Fewer modes since we have limited data
        "width": 64,         # Narrower network to match data size
        "device": device,
    }

    training_config = {
        'batch_size': 5,
        'learning_rate': 0.001,    # Slightly lower learning rate
        'epochs': 2000,           # More epochs since we have patience
        'weight_decay': 1e-6,     # Stronger regularization
        'optimizer': 'AdamW',
        'patience': 200,
        'freq_print': 1,
        'training_mode': 'all2all',
        'grad_clip': 0.5          # Tighter gradient clipping
    }

    naming_config = {
        **model_config,
        'learning_rate': training_config['learning_rate'],
        'training_mode': training_config['training_mode']
    }
    
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/custom_fno/time") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)

    # Create datasets and dataloaders
    train_dataset = PDEDataset(
        data_path="data/train_sol.npy",
        timesteps=5,
        which="training",
        training_samples=64,
        training_mode=training_config['training_mode']
    )
    val_dataset = PDEDataset(
        data_path="data/train_sol.npy",
        timesteps=5,
        which="validation",
        training_mode=training_config['training_mode']
    )
    test_dataset = PDEDataset(
        data_path="data/train_sol.npy",
        timesteps=5,
        which="test",
        training_mode=training_config['training_mode']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)

    # Instantiate model on the specified device
    model = FNO1d(**model_config).to(device)  # or model = FNO1d(**model_config)

    # Train model (make sure your train_model function also moves inputs/targets to device)
    trained_model, training_history = train_model(
        model,
        train_loader,
        val_loader, 
        training_config,
        checkpoint_dir,
        device  # Pass the device to train_model so it can move batches to GPU
    )

    print("Plotting training histories...")
    plot_training_history(experiment_dir=checkpoint_dir)

    print(f"Time-dependent FNO training completed. Model saved in {checkpoint_dir}")
    print(f"Best validation loss: {training_history['best_val_loss']:.6f} "
          f"at epoch {training_history['best_epoch']}")

if __name__ == "__main__":
    main()
