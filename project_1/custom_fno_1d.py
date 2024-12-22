import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
from pathlib import Path
from training import prepare_data, train_model

class SpectralConv1d(nn.Module):
    """The FNO1d uses SpectralConv1d as its crucial part."""
    def __init__(self, in_channels, out_channels, modes1, use_bn=True):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
            
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]

        # using real-valued FFT for real-valued input discretized state x
        x_ft = torch.fft.rfft(x)

        # Batched matrix multiplication for the actual modes we specified
        # Modes in Fourier space reflect frequency
        # - lower mode => smaller frequency => longer periods (global information)
        # - those with high frequency get cut off (normally local information)
        actual_modes = min(self.modes1, x.size(-1)//2 + 1)        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                           device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :actual_modes] = self.compl_mul1d(x_ft[:, :, :actual_modes], 
                                                      self.weights1[:, :, :actual_modes])
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        if self.use_bn:
            x = self.bn(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, depth, use_norm=False):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.depth = depth
        self.use_norm = use_norm
        
        self.linear_p = nn.Linear(2, self.width)
        if self.use_norm:
            self.norm_p = nn.LayerNorm(self.width)
        
        self.fourier_layers = nn.ModuleList([
            self._make_fourier_layer() for _ in range(self.depth)
        ])
        
        self.linear_q = nn.Linear(self.width, self.width)
        if self.use_norm:
            self.norm_q = nn.LayerNorm(self.width)
        
        self.output_layer = nn.Linear(self.width, 1)
        self.activation = nn.Softplus()
    
    def _make_fourier_layer(self):
        return nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes1),
            nn.Conv1d(self.width, self.width, 1),
            nn.LayerNorm(self.width) if self.use_norm else nn.Identity()
        ])
    
    def forward(self, x):
        x = self.linear_p(x)
        if self.use_norm:
            x = self.norm_p(x)
        x = self.activation(x)
        
        x = x.permute(0, 2, 1)
        
        for spect, conv, norm in self.fourier_layers:
            spectral_out = spect(x)
            conv_out = conv(x)
            combined = spectral_out + conv_out
            if self.use_norm:
                combined = combined.permute(0, 2, 1)
                combined = norm(combined)
                combined = combined.permute(0, 2, 1)
            x = self.activation(combined)
        
        x = x.permute(0, 2, 1)
        x = self.linear_q(x)
        if self.use_norm:
            x = self.norm_q(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Configuration
    # 9.47%
    # config = {
    #     'n_train': 64,
    #     # 'batch_size': 5, => 7.27%
    #     # 'batch_size': 2, => 8.60%
    #     # 'batch_size': 8, => 17.51%
    #     'batch_size': 5,

    #     # 'modes': 32,
    #     'modes': 10,

    #     # 'width': 64,
    #     # 'width': 128, => 9.28% 
    #     'width': 128,
    #     # 'depth': 2, => 9.12%
    #     'depth': 1, # => 7.43%
    #     # 'depth': 10, => 14.92%
    #     'learning_rate': 0.001,
    #     'epochs': 500,
    #     # 'step_size': 50, => 9.14%
    #     # 'step_size': 100, => 8.57%
    #     'step_size': 100,
    #     # 'step_size': 250, => 11.18%
    #     # 'step_size': 500, => 11.18%

    #     # 'gamma': 0.5,
    #     # 'gamma': 0.1, => 9.47% 
    #     'gamma': 0.1,

    #     'patience': 50,
    #     'freq_print': 1
    # }

    # Modes - 16
    # depth 4: 10.32%
    # depth 3: 8.76%
    # depth 2: 6.83%
    # depth 1: 8.93%

    # Modes - 17
    # depth 2: 6.40%

    # Modes - 18
    # depth 2: 6.65%

    # Modes - 19
    # depth 2: 6.2%

    # Modes - 25
    # depth 2: 6.07%

    # Modes - 35
    # depth 2: 6.91%
    config = {
        'n_train': 64,
        'batch_size': 5,
        'modes': 25,
        'width': 64,
        'depth': 2, # => 7.43%
        'learning_rate': 0.001,
        'epochs': 500,
        'step_size': 100,
        'gamma': 0.1,
        'patience': 50,
        'freq_print': 1
    }    
    
    # Prepare data
    training_set, testing_set, test_data = prepare_data("data/train_sol.npy", 
                                                      config['n_train'], 
                                                      config['batch_size'])
    
    # Initialize model
    model = FNO1d(modes=config['modes'], width=config['width'], depth=config['depth'], use_norm=False)
    
    # Train model
    trained_model = train_model(model, training_set, testing_set, config, 
                              "checkpoints/custom_fno")
    
    print("Training completed. Model saved in checkpoints/custom_fno/best_model.pth")

if __name__ == "__main__":
    main()