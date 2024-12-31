"""
Code reference and inspiration taken from:

- FNO implementation:
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/_OtherModels/FNOModules.py

- Time-dependent batch normalization layer (FILM):
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/CNO2d_time_dependent_%26_foundation_model/CNO_timeModule_CIN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FILM(nn.Module):
    """Time-conditional normalization layer, only used in FNO when time-dependent is set"""
    def __init__(self, channels, use_bn=True):
        super(FILM, self).__init__()
        self.channels = channels
        self.inp2scale = nn.Linear(in_features=1, out_features=channels, bias=True)
        self.inp2bias = nn.Linear(in_features=1, out_features=channels, bias=True)
        
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

    def forward(self, x, time):
        """
        Apply FiLM (Feature-wise Linear Modulation) conditioning to input features.
        
        Args:
           x (torch.Tensor): Input feature tensor of shape [batch_size, channels, *spatial_dims]
           time (torch.Tensor): Conditioning tensor of shape [batch_size]
        
        Returns:
           torch.Tensor: Modulated features with same shape as input
        """
        time = time.reshape(-1, 1).type_as(x)  # [B] -> [B,1]
        
        # Apply normalization
        x = self.norm(x)  # [B,C,L]
        
        # Get scale and bias terms
        scale = self.inp2scale(time)     # [B, 1] -> [B, C]
        bias = self.inp2bias(time)       # [B, 1] -> [B, C]
        
        # Adjust dimensions for broadcasting
        scale = scale.unsqueeze(-1)  # [B, C] → [B, C, 1]
        bias = bias.unsqueeze(-1)    # [B, C] → [B, C, 1]
        
        return x * scale + bias

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
    def __init__(self, modes, width, depth, device="cuda", nfun=1, padding_frac=1/4, time_dependent=False):
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
        super(FNO1d, self).__init__()
        
        self.modes = modes                
        self.width = width                
        self.depth = depth
        self.padding_frac = padding_frac
        self.time_dependent = time_dependent
        
        self.fc0 = nn.Linear(nfun + 1, self.width) # +1 for x_grid
        
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
        
        # Time-conditional normalization (only if time_dependent)
        if time_dependent:
            print(f"Time dependent is set to {time_dependent}")
            self.film_list = nn.ModuleList([
                FILM(self.width, use_bn=True) for _ in range(self.depth)
            ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.activation = nn.GELU()

        self.to(device)  # Move model to specified device

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channel]
            
        Returns:
            Output tensor of shape [batch_size, sequence_length, 1]
        """        
        u_start = x[..., 0].unsqueeze(-1)
        v_start = x[..., 1].unsqueeze(-1)
        x_grid  = x[..., 2].unsqueeze(-1)

        # dt will not be used if time-dependent is not set True
        dt = x[..., 3].unsqueeze(-1)  # Keep as [batch]
        
        x = torch.cat((u_start, v_start, x_grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # Now shape is [batch, width, sequence_length]
        
        # Apply padding
        x_padding = int(round(x.shape[-1] * self.padding_frac))
        x = F.pad(x, [0, x_padding])
        
        # Apply Fourier layers with residual connections and bias
        for i in range(self.depth):
            # Store input for residual connection
            x_input = x
            # Fourier integral operator K^(l)
            x1 = self.spectral_list[i](x)  # Shape: [batch, width, sequence_length]
            # Residual connection W^(l)
            x2 = self.w_list[i](x_input.transpose(1, 2)).transpose(1, 2)
            # Expand b to match the sequence length
            b_expanded = self.b_list[i].expand(-1, -1, x1.size(-1))
            
            # Combine with bias: K^(l) + W^(l) + b^(l)
            x = x1 + x2 + b_expanded
            
            # Apply time-conditional normalization if time-dependent
            if self.time_dependent:
                # tensor([[a], [b], [c], [d], [e]])
                dt = dt[:, 0].unsqueeze(-1)  # or dt[:, 0].view(-1, 1)
                x = self.film_list[i](x, dt)
            
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