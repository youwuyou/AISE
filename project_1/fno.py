"""
Code reference and inspiration taken from:

- FNO implementation:
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/_OtherModels/FNOModules.py

- Time-dependent batch normalization layer:
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/517b0ee78a97ed2a7883470a418d2c65eae68d3d/CNO2d_time_dependent_%26_foundation_model/CNO_timeModule_CIN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import FILM, SpectralConv1d

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
            self.film_list = nn.ModuleList([
                FILM(self.width, use_bn=True) for _ in range(self.depth)
            ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.activation = nn.GELU()

        self.to(device)  # Move model to specified device

    def forward(self, x, t=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channel]
            t: Optional time tensor of shape [batch_size] for time-dependent model
            
        Returns:
            Output tensor of shape [batch_size, sequence_length, 1]
        """
        if self.time_dependent:
            # TODO: this data format is wrong!
            if t is None:
                raise ValueError("Time tensor 't' must be provided when model is time-dependent")
            # Time-dependent forward pass
            x = x.permute(0, 2, 1)  # Change to [batch, width, sequence_length]
            x = self.fc0(x)         # Apply lifting layer
        else:
            # Time-independent forward pass
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
            x2 = self.w_list[i](x_input.transpose(1, 2))  # Input: [batch, sequence_length, width]
            x2 = x2.transpose(1, 2)  # Back to [batch, width, sequence_length]
            
            # Expand b to match the sequence length
            b_expanded = self.b_list[i].expand(-1, -1, x1.size(-1))
            
            # Combine with bias: K^(l) + W^(l) + b^(l)
            x = x1 + x2 + b_expanded
            
            # Apply time-conditional normalization if time-dependent
            if self.time_dependent:
                x = self.film_list[i](x, t)
            
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