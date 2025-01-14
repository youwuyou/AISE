"""
This module contains the backbone of the used mini neural foundation model based on FNO, which is
very similar to the project_1/fno.py, with small modifications.
    - we used general-purposed conditional layer FiLM now for both ɛ and t
    - previously we used it for t only

Code reference and inspiration taken from:

- FNO implementation:
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/_OtherModels/FNOModules.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    The FNOBlock uses SpectralConv1d as its crucial part, 
        - implements the Fourier integral operator in a layer
            F⁻¹(R⁽ˡ⁾∘F)(u)
        - uses FFT, linear transform, and inverse FFT, applicable to equidistant mesh 
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

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
        effective_modes = min(self.modes, x.size(-1) // 2 + 1)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                        device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :effective_modes] = self.compl_mul1d(x_ft[:, :, :effective_modes], 
                                                    self.weights1[:,:,:effective_modes])
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width, depth=3, padding_frac=1/4):
        """
        The overall network 𝒢_θ(u). It contains [depth] layers of the Fourier layers.
        Each layer implements:
        u_(l+1) = σ(K^(l)(u_l) + W^(l)u_l + b^(l))
        where:
        - K^(l) is the Fourier integral operator F⁻¹(R⁽ˡ⁾∘F)
        - W^(l) is the residual connection
        - b^(l) is the bias term
        
        Complete architecture 
            - Lifting and projecting done within AllenCahnFNO, unlike in project 1 done in FNO directly
            - Apply [depth] layers of Fourier integral operators with residual connections
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.padding_frac = padding_frac

        # Fourier layers
        self.spectral_list = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)
        ])

        # Linear residual connections
        self.w_list = nn.ModuleList([
            nn.Linear(self.width, self.width, bias=False) for _ in range(self.depth)
        ])

        # Bias terms
        self.b_list = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.width, 1)) for _ in range(self.depth)
        ])

        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size * n_steps, width, x_size]
        Returns:
            torch.Tensor: Output tensor after applying spectral convolution and residual connections
        """
        # Apply padding
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = F.pad(x, [0, x_padding])

        # Apply Fourier layers with residual connections and bias
        for i in range(self.depth):
            # Store input for residual connection
            x_input = x
            # Fourier integral operator K^(l)
            x1 = self.spectral_list[i](x)  # [batch_size * n_steps, width, x_size]
            # Residual connection W^(l)
            x2 = self.w_list[i](x_input.transpose(1, 2)).transpose(1, 2)
            # Expand b to match the sequence length
            b_expanded = self.b_list[i].expand(-1, -1, x1.size(-1))
            
            # Combine with bias: K^(l) + W^(l) + b^(l)
            x = x1 + x2 + b_expanded
                        
            # Apply activation (except for last layer)
            if i != self.depth - 1:
                x = self.activation(x)
        
        # Remove padding
        x = x[..., :-x_padding]
        
        return x  # Shape should be [batch_size * n_steps, width, x_size]

class FILM(nn.Module):
    """
    General conditional normalization layer using FILM (Feature-wise Linear Modulation).
    Applies conditioning based on both epsilon (ε) and time (t).
    """
    def __init__(self, channels, in_features=2, use_bn=True):
        super().__init__()
        self.channels = channels
        self.inp2scale = nn.Linear(in_features=in_features, out_features=channels, bias=True)  # 2 features: ε and t
        self.inp2bias = nn.Linear(in_features=in_features, out_features=channels, bias=True)
        
        # Initialize to identity mapping
        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(1)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)
        
        # Use BatchNorm if specified, otherwise Identity
        if use_bn:
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x, eps, t):
        """
        Apply FILM (Feature-wise Linear Modulation) conditioning to input features.
        
        Args:
           x (torch.Tensor): Input feature tensor of shape [batch_size * n_steps, channels, x_size]
           eps (torch.Tensor): Epsilon tensor of shape [batch_size * n_steps, 1]
           t (torch.Tensor): Time tensor of shape [batch_size * n_steps, 1]
        
        Returns:
           torch.Tensor: Modulated features with same shape as input
        """
        # Concatenate epsilon and time features
        feature = torch.cat([eps, t], dim=1)  # [batch_size * n_steps, 2]
        
        # Apply normalization
        x = self.norm(x)  # [batch_size * n_steps, channels, x_size]
        
        # Get scale and bias terms
        scale = self.inp2scale(feature)     # [batch_size * n_steps, channels]
        bias = self.inp2bias(feature)       # [batch_size * n_steps, channels]
        
        # Adjust dimensions for broadcasting
        scale = scale.unsqueeze(-1)  # [batch_size * n_steps, channels, 1]
        bias = bias.unsqueeze(-1)    # [batch_size * n_steps, channels, 1]
        
        return x * scale + bias

class AllenCahnFNO(nn.Module):
    def __init__(self, modes=16, width=64, depth=4, device="cuda"):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth

        # Input layer: maps input scalar to hidden channels
        self.input_layer = nn.Linear(1, self.width)

        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(in_channels=self.width, out_channels=self.width, modes=self.modes, width=self.width, depth=depth) for _ in range(self.depth)
        ])

        # FILM layers for conditioning on ε and t
        self.FILM_layers = nn.ModuleList([
            FILM(self.width, use_bn=True) for _ in range(self.depth)
        ])

        # Activation function
        self.activation = nn.GELU()

        # Output layer: maps hidden channels back to scalar output
        self.output_layer = nn.Linear(self.width, 1)

        self.to(device)  # Move entire model to specified device

    def forward(self, x, eps, t):
        batch_size, x_size = x.shape
        n_steps = t.shape[1]

        # Expand x and eps to match time steps
        x_expanded = x.unsqueeze(1).repeat(1, n_steps, 1)  # [batch_size, n_steps, x_size]
        eps_expanded = eps.repeat(1, n_steps)             # [batch_size, n_steps]
        t_expanded = t                                    # [batch_size, n_steps]

        # Flatten the batch and time dimensions for processing
        x_flat = x_expanded.reshape(batch_size * n_steps, x_size)    # [batch_size * n_steps, x_size]
        eps_flat = eps_expanded.reshape(batch_size * n_steps, 1)     # [batch_size * n_steps, 1]
        t_flat = t_expanded.reshape(batch_size * n_steps, 1)         # [batch_size * n_steps, 1]

        # Input layer
        u = self.input_layer(x_flat.unsqueeze(-1))  # [batch_size * n_steps, x_size, width]
        u = u.permute(0, 2, 1)                      # [batch_size * n_steps, width, x_size]

        # Iterate through FNO layers with FILM conditioning
        for layer_idx in range(self.depth):
            # Apply FNO layer
            u = self.fno_layers[layer_idx](u)       # [batch_size * n_steps, width, x_size]
            u = self.activation(u)                   # Non-linear activation
            u = self.FILM_layers[layer_idx](u, eps_flat, t_flat)  # [batch_size * n_steps, width, x_size]

        # Output layer
        u = u.permute(0, 2, 1)                      # [batch_size * n_steps, x_size, width]
        u = self.output_layer(u)                    # [batch_size * n_steps, x_size, 1]
        u = u.squeeze(-1)                           # [batch_size * n_steps, x_size]
        u = u.reshape(batch_size, n_steps, x_size)
        return u

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Model initialization
    model = AllenCahnFNO(modes=16, width=128, depth=4, device=device)

    # Sample data
    batch_size, x_size = 32, 128
    u0 = torch.randn(batch_size, x_size, device=device)
    ut = u0.unsqueeze(1).expand(-1, 4, -1)

    eps = torch.randn(batch_size, 1, device=device)
    t = torch.linspace(0, 1, 4, device=device)[None].expand(batch_size, -1)  # Move to device

    # Forward pass
    ut_pred = model(u0, eps, t)  # Should return (batch_size, n_steps, x_size)
    print(f"Input: {u0}")
    print(f"u(x, t):   {ut}")
    print(f"Output: {ut_pred}")

    print(f"Input shape: {u0.shape}")
    print(f"Output shape: {ut_pred.shape}")
    print(f"Error: {torch.abs(ut - ut_pred)}")