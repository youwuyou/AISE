"""
Temporal place for unchanged routine!"""


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
