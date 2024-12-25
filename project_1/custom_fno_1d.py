import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
from pathlib import Path
from training import train_model, get_experiment_name, save_config, prepare_data




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

        # TODO: now trying out identity as activation function
        self.activation = nn.Softplus()
        # self.activation = nn.SELU() # worse than Softplus


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

"""
The best performing configurations achieved error rates around 6-7%, with the optimal settings being:

Modes: 25 (6.07% error) with depth=2
Batch size: 5 (7.27% error), better than 2 (8.60%) or 8 (17.51%)
Depth: 2 appears optimal for most configurations, with depth=1 giving 7.43% and depth=10 leading to worse performance (14.92%)
Width: Both 64 and 128 performed similarly (around 9%)
Step size: 100 performed best (8.57%), with larger values (250, 500) increasing error to 11.18%
Gamma: 0.1 was used consistently

The most sensitive parameters appear to be:

Number of modes (ranging from 6.07% to 9.47% error)
Depth (ranging from 6.83% to 14.92% error)
Batch size (ranging from 7.27% to 17.51% error)

The final configuration shown uses these insights with modes=25, depth=2, and batch_size=5, which aligns with the better performing settings discovered during testing.
"""

def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Model configuration
    model_config = {
        'modes': 25, # 5.88% resolution 64, 9.68% resolution 32
        # 'modes': 20, # 7.08% resolution 64, 9.98% resolution 32
        # 'modes': 15, # 6.45% resolution 64, 9.40% resolution 32
        # 'modes': 10, # 9.04% resolution 64, 10.82% resolution 32
        'width': 64,
        'depth': 2,
        'use_norm': False,
        'model_type': 'custom'  # Add model type identifier
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
    checkpoint_dir = Path("checkpoints/custom_fno") / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full configuration
    config = {
        'model_config': model_config,
        'training_config': training_config
    }
    save_config(config, checkpoint_dir)
    
    # Prepare data (note: custom FNO doesn't use library=True flag)
    training_set, testing_set, test_data = prepare_data(
        "data/train_sol.npy",
        training_config['n_train'],
        training_config['batch_size']
    )
    
    # Initialize model
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
    main()