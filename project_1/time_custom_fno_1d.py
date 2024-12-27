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
save_config
)

from visualization import plot_training_history


class FILM(nn.Module):
    """
    Time-conditional normalization layer
        - Implements time-dependent feature modulation
        - Combines batch normalization with time-based scaling and shifting
        - Similar to FiLM (Feature-wise Linear Modulation) but conditioned on time
    """
    def __init__(self, channels, use_bn=True):
        super(FILM, self).__init__()
        self.channels = channels
        
        # Scale and bias networks for time conditioning
        self.inp2scale = nn.Linear(1, channels, bias=True)
        self.inp2bias = nn.Linear(1, channels, bias=True)
        
        # Initialize to identity mapping for stable training
        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(1)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)
        
        # Optional batch normalization
        if use_bn:
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x, time):
        """
        Apply time-conditional normalization to input features.
        
        Args:
            x (torch.Tensor): Input features, shape [B, C, X]
            time (torch.Tensor): Time conditioning, shape [B]
            
        Returns:
            torch.Tensor: Modulated features, same shape as x
        """
        # 1) Apply batch normalization if enabled
        x = self.norm(x)
        
        # 2) Process time information
        time = time.reshape(-1, 1).type_as(x)  # [B] -> [B,1]
        
        # 3) Get modulation parameters from time
        scale = self.inp2scale(time)     # [B, C]
        bias  = self.inp2bias(time)      # [B, C]
        
        # 4) Reshape for broadcasting
        scale = scale.unsqueeze(-1)      # [B, C, 1]
        bias  = bias.unsqueeze(-1)       # [B, C, 1]
        
        return x * scale + bias


class SpectralConv1d(nn.Module):
    """
    The FNO1d uses SpectralConv1d as its crucial part,
        - implements the Fourier integral operator in a layer
        - uses FFT, linear transform, and inverse FFT
        - modified to work with time-dependent features
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        if not isinstance(modes1, int) or modes1 <= 0:
            raise ValueError(f"modes1 must be a positive integer, got {modes1}")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, dtype=torch.cfloat
        ))

    def compl_mul1d(self, input, weights):
        """
        Complex multiplication in 1D
        (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        """
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x):
        """
        Forward pass:
            1) Compute Fourier coefficients
            2) Multiply relevant Fourier modes
            3) Transform back to physical space
        """
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        effective_modes = min(self.modes1, x.size(-1) // 2 + 1)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :effective_modes] = self.compl_mul1d(
            x_ft[:, :, :effective_modes], 
            self.weights1[:, :, :effective_modes]
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, depth, device="cpu", nfun=1, padding_frac=1/4):
        super(FNO1d, self).__init__()
        """
        Time-dependent FNO1d network.
        The overall network. It contains [depth] layers of the Fourier layer.
        1) Lift the input to the desired channel dimension by self.fc0
        2) [depth] layers of Fourier integral operators with time-conditional normalization
        3) Project from the channel space to the output space by self.fc1 and self.fc2
        """
        self.modes = modes                
        self.width = width                
        self.depth = depth                
        self.padding_frac = padding_frac  
        
        # Lifting layer (assuming input has shape [B, 2, X] => x, time)
        self.fc0 = nn.Linear(2, self.width)
        
        # Fourier integral operator layers
        self.spectral_list = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)
        ])
        
        # Residual layers
        self.w_list = nn.ModuleList([
            nn.Linear(self.width, self.width, bias=False) for _ in range(self.depth)
        ])
        
        # Time-conditional normalization layers
        self.film_list = nn.ModuleList([
            FILM(self.width, use_bn=True) for _ in range(self.depth)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.activation = nn.GELU()
        self.to(device)

    def forward(self, x, t):
        """
        Input:
            x: [batch_size, 2, spatial_size]
            t: [batch_size] or [batch_size, 1]
        Output:
            [batch_size, 1, spatial_size]
        """
        # Reshape input for lifting layer
        x = x.permute(0, 2, 1)  # [batch_size, spatial_size, 2]
        
        # Lifting to higher dimension
        x = self.fc0(x)        # [batch_size, spatial_size, width]
        x = x.permute(0, 2, 1) # [batch_size, width, spatial_size]
        
        # Padding
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = F.pad(x, [0, x_padding])
        
        # Apply the FNO layers
        for i, (spectral, w_linear, film) in enumerate(zip(self.spectral_list, self.w_list, self.film_list)):
            x_input = x
            
            # Fourier operator
            x1 = spectral(x)
            
            # Residual connection
            x2 = w_linear(x_input.transpose(1, 2))  # [B, X, width]
            x2 = x2.transpose(1, 2)                # [B, width, X]
            
            # Combine
            x = x1 + x2
            
            # Time-conditional normalization
            x = film(x, t)
            
            if i != self.depth - 1:
                x = self.activation(x)
        
        # Remove padding
        x = x[..., :-x_padding]
        
        # Projection layers
        x = x.permute(0, 2, 1)   # [batch_size, spatial_size, width]
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Final shape [batch_size, spatial_size, 1] => transpose if you need [B, 1, X]
        x = x.permute(0, 2, 1)
        return x

    def print_size(self):
        """Prints the total number of parameters in the model"""
        nparams = 0
        for param in self.parameters():
            nparams += param.numel()
        print(f'Total number of model parameters: {nparams}')
        return nparams


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model_config = {
        "depth": 6,          # Increase depth
        "modes": 48,         # More modes for complex dynamics
        "width": 256         # Wider network
    }

    training_config = {
        'batch_size': 32,    # Larger batch size
        'learning_rate': 5e-5, # Lower learning rate
        'epochs': 1000,      # More epochs
        'step_size': 300,    # Even longer LR period
        'gamma': 0.5,
        'patience': 60,      # More patience for convergence
        'freq_print': 1,
        'training_mode': 'all2all'
    }

    # Direct Inference at multiple timesteps
    # Average relative L2 error at t=0.25: 36.4949%
    # Average relative L2 error at t=0.50: 79.6020%
    # Average relative L2 error at t=0.75: 118.8351%
    # Average relative L2 error at t=1.00: 170.9276%

    # Auto regressive at multiple timesteps
    # Average relative L2 error at t=0.25: 36.4949%
    # Average relative L2 error at t=0.50: 106.4158%
    # Average relative L2 error at t=0.75: 186.2448%
    # Average relative L2 error at t=1.00: 301.3198%

    # Testing OOD at t = 1.0
    # Average relative L2 error on OOD data: 180.3255%

    # model_config = {
    #     "depth": 4,          # Increase from 2 to 4
    #     "modes": 32,         # Slight increase
    #     "width": 128         # Double the width
    # }

    # training_config = {
    #     'batch_size': 16,    # Increase from 5
    #     'learning_rate': 1e-4, # Decrease from 1e-3
    #     'epochs': 500,       # More epochs
    #     'step_size': 200,    # Longer learning rate period
    #     'gamma': 0.5,        # Less aggressive decay
    #     'patience': 40,      # Shorter patience,
    #     'freq_print': 1,
    #     'training_mode': 'all2all'  # or 'one-at-a-time'
    # }

    # Direct Inference at multiple timesteps
    # Average relative L2 error at t=0.25: 68.1075%
    # Average relative L2 error at t=0.50: 111.8394%
    # Average relative L2 error at t=0.75: 135.1795%
    # Average relative L2 error at t=1.00: 178.0903%

    # Auto regressive at multiple timesteps
    # Average relative L2 error at t=0.25: 68.1075%
    # Average relative L2 error at t=0.50: 237.4745%
    # Average relative L2 error at t=0.75: 558.5176%
    # Average relative L2 error at t=1.00: 1327.8747%
    # model_config = {
    #     "depth": 2,
    #     "modes": 30,
    #     "width": 64
    # }
    
    # training_config = {
    #     'batch_size': 5,
    #     'learning_rate': 0.001,
    #     'epochs': 400,
    #     'step_size': 100,
    #     'gamma': 0.1,
    #     'patience': 50,
    #     'freq_print': 1,
    #     'training_mode': 'all2all'  # or 'one-at-a-time'
    # }
    
    naming_config = {
        **model_config,
        'learning_rate': training_config['learning_rate'],
        'training_mode': training_config['training_mode']  # Add to experiment name
    }
    
    experiment_name = get_experiment_name(naming_config)
    checkpoint_dir = Path("checkpoints/custom_fno/time") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_config': model_config,
        'training_config': training_config
    }

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

    model = FNO1d(**model_config)
    trained_model, training_history = train_model(
        model,
        train_loader,
        val_loader,
        training_config,
        checkpoint_dir
    )

    config['training_history'] = training_history
    save_config(config, checkpoint_dir)

    print("Plotting training histories...")
    plot_training_history(experiment_dir=checkpoint_dir)

    print(f"Time-dependent FNO training completed. Model saved in {checkpoint_dir}")
    print(f"Best validation loss: {training_history['best_val_loss']:.6f} "
          f"at epoch {training_history['best_epoch']}")

if __name__ == "__main__":
    main()
