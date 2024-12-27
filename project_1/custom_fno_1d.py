"""
Code reference and inspiration taken from:

https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/_OtherModels/FNOModules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from training import train_model, get_experiment_name, save_config, prepare_data


class SpectralConv1d(nn.Module):
    """
    The FNO1d uses SpectralConv1d as its crucial part, 
        - implements the Fourier integral operator in a layer
            F⁻¹(R⁽ˡ⁾∘F)(u)
        - uses FFT, linear transform, and inverse FFT, applicable to equidistant mesh 
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        if not isinstance(modes1, int) or modes1 <= 0:
            raise ValueError(f"modes1 must be a positive integer, got {modes1}")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        """
        Complex multiplication in 1D
        (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        """
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x):
        """
            1) Compute Fourier coefficients
            2) Multiply relevant Fourier modes
            3) Transform the data to physical space
            HINT: Use torch.fft library torch.fft.rfft        
        """
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        
        # Use min to limit modes to what's available
        effective_modes = min(self.modes1, x.size(-1) // 2 + 1)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                        device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :effective_modes] = self.compl_mul1d(x_ft[:, :, :effective_modes], 
                                                    self.weights1[:,:,:effective_modes])
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, depth, device="cuda", nfun=1, padding_frac=1/4):  # Changed default device
       super(FNO1d, self).__init__()
       """
       The overall network 𝒢_θ(u). It contains [depth] layers of the Fourier layers.
       Each layer implements:
       u_(l+1) = σ(K^(l)(u_l) + W^(l)u_l + b^(l))
       where:
       - K^(l) is the Fourier integral operator F⁻¹(R⁽ˡ⁾∘F)
       - W^(l) is the residual connection
       - b^(l) is the bias term
       
       Complete architecture:
       1. Lift the input to the desired channel dimension by P (self.fc0)
       2. Apply [depth] layers of Fourier integral operators with residual connections
       3. Project from the channel space to the output space by Q (self.fc1 and self.fc2)

       input: the solution of the initial condition and location (a(x), x)
       input shape: (batchsize, x=s, c=2)
       output: the solution of a later timestep
       output shape: (batchsize, x=s, c=1)
       """
       self.modes = modes                
       self.width = width                
       self.depth = depth                
       self.padding_frac = padding_frac  
       
       self.fc0 = nn.Linear(nfun + 1, self.width)
       
       self.spectral_list = nn.ModuleList([
           SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)
       ])
       
       self.w_list = nn.ModuleList([
           nn.Linear(self.width, self.width, bias=False) for _ in range(self.depth)
       ])
       
       self.b_list = nn.ParameterList([
           nn.Parameter(torch.zeros(1, self.width, 1)) for _ in range(self.depth)
       ])
       
       self.fc1 = nn.Linear(self.width, 128)
       self.fc2 = nn.Linear(128, 1)

       self.activation = nn.GELU()

       self.to(device)  # Move model to specified device

    def forward(self, x):
        # Initial lifting layer P
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # Now shape is [batch, width, sequence_length]
        
        # Apply padding
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = F.pad(x, [0, x_padding])
        
        # Apply Fourier layers with residual connections and bias
        for i, (spectral, w, b) in enumerate(zip(self.spectral_list, self.w_list, self.b_list)):
            # Store input for residual connection
            x_input = x
            
            # Fourier integral operator K^(l)
            x1 = spectral(x)  # Shape: [batch, width, sequence_length]
            
            # Residual connection W^(l)
            x2 = w(x_input.transpose(1, 2))  # Input: [batch, sequence_length, width]
            x2 = x2.transpose(1, 2)  # Back to [batch, width, sequence_length]
            
            # Expand b to match the sequence length
            b_expanded = b.expand(-1, -1, x1.size(-1))
            
            # Combine with bias: K^(l) + W^(l) + b^(l)
            x = x1 + x2 + b_expanded
            
            # Apply activation (except for last layer)
            if i != self.depth - 1:
                x = self.activation(x)
        
        # Remove padding
        x = x[..., :-x_padding]
        
        # Final projection layers Q
        x = x.permute(0, 2, 1)  # Back to [batch, sequence_length, width]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x  # Shape should be [batch_size, sequence_length, 1]

    def print_size(self):
        nparams = 0
        nbytes = 0
        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()
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
   
   # Model configuration
   # checkpoints/custom_fno/fno_m30_w64_d4_lr0.001_20241227_190139
   # 11.99% resolution 32; 4.92% resolution 64
   model_config = {
       "depth": 4,
       "modes": 30,
       "width": 64,
       "model_type": "custom",
       "device": device
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
       'freq_print': 1,
       'device': device
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
   
   # Prepare data
   training_set, testing_set, test_data = prepare_data(
       "data/train_sol.npy",
       training_config['n_train'],
       training_config['batch_size'],
       device=device
   )
   
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
   main()