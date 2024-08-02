import torch
import torch.nn as nn
import torch.fft
from lightning_utils import *

# Verified to work 7/19/24
class MORLayer2D(BasicLightningRegressor):
    """ A single 2D MOR operator layer. """
    def __init__(self, k_modes=32, activation=nn.SiLU):
        super().__init__()
        vars(self).update(locals()); del self.self
        self.g_mode_params = nn.Parameter(torch.randn(k_modes, k_modes//2+1, 2))
        self.h_mlp=CNN(k_size=1, activation=activation)
        # Define the weights in the Fourier domain (complex values)

    def forward(self, u):
        u = torch.as_tensor(u).to(self.device)

        # Apply point-wise MLP nonlinearity h(u)
        h_u = self.h_mlp(u)

        # Apply Fourier transform (last 2 dims need to be x&y!)
        u_fft = torch.fft.rfft2(h_u)

        # Convert to complex dtype & pad the g_mode_params (this will drop extra modes)
        g = torch.view_as_complex(self.g_mode_params)
        g_padded = torch.zeros(u_fft.shape[-2],u_fft.shape[-1], dtype=g.dtype, device=g.device)
        g_padded[:g.shape[0], :g.shape[1]] = g
        # ChatGPT says this is correct b/c apparently for rfft2 the lowest modes are in the top-left corner only.

        if len(u_fft.shape)==3:
            g_padded = g_padded[None]

        # Apply learned weights in the Fourier domain
        u_fft = u_fft * g_padded

        # Apply inverse Fourier transform
        u_ifft = torch.fft.irfft2(u_fft)

        # Return real part as output
        return u_ifft

# Verified to work 7/19/24
class MOR_Layer(BasicLightningRegressor):
    """ A single Nd MOR operator layer. """
    def __init__(self, in_channels=1, out_channels=1, k_modes=32, ndims=2, **kwd_args):
        super().__init__()
        vars(self).update(locals()); del self.self
        g_shape = [in_channels, out_channels]+[k_modes]*(ndims-1) + [k_modes//2+1, 2]
        self.g_mode_params = nn.Parameter(torch.randn(*g_shape))
        self.h_mlp=CNN(in_channels,in_channels,k_size=1, conv_dims=ndims, **kwd_args)
        # Define the weights in the Fourier domain (complex values)

    def forward(self, u):
        u = torch.as_tensor(u).to(self.device)

        # Apply point-wise MLP nonlinearity h(u)
        h_u = self.h_mlp(u) # keeps channel size the same!

        # Apply Fourier transform (last 2 dims need to be x&y!)
        u_fft = torch.fft.rfftn(h_u, [-1]*self.ndims) # should FFT the last self.ndims modes

        # Convert to complex dtype & pad the g_mode_params (this will drop extra modes)
        g = torch.view_as_complex(self.g_mode_params)
        g_padded_shape = list(g.shape[:-self.ndims])+list(u_fft.shape[-self.ndims:]) # fuse shapes
        g_padded = torch.zeros(*g_padded_shape, dtype=g.dtype, device=g.device)

        # the first part selects all of the input and output channel dimensions
        low_pass_slices = [slice(None)]*2 + [slice(s_i) for s_i in g.shape]
        g_padded[low_pass_slices] = g
        # ChatGPT says this is correct b/c apparently for rfft2 the lowest modes are
        # concentrated near the origin (e.g. top-left corner for 2d)

        if len(u_fft.shape)>len(g_padded.shape):
            g_padded = g_padded[None] # add batch dimension (if needed)

        # Apply learned weights in the Fourier domain
        u_fft = torch.einsum('ij...,ijk...->ik...', u_fft, g_padded) # einsum does channel reduction
        #u_fft = u_fft * g_padded

        # Apply inverse Fourier transform
        u_ifft = torch.fft.irfftn(u_fft, [-1]*self.ndims) # should IFFT the last self.ndims modes

        # Return real part as output
        return u_ifft

class MOR_Operator(BasicLightningRegressor):
    """
    Essentially a stack of MORLayer-Nd layers + skip connections.
    Without skip-connections this operator doesn't work at all
    (assuming multiple layers), b/c of 1 channel bottleneck.
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32,
                 n_layers=4, **kwd_args):
        super().__init__()
        self.layers = nn.ModuleList([MOR_Layer(in_channels, hidden_channels, **kwd_args)] +
            [MOR_Layer(hidden_channels, hidden_channels, **kwd_args) for i in range(n_layers-2)]+
            [MOR_Layer(hidden_channels, out_channels, **kwd_args)])
    def forward(self, X):
        for layer in self.layers[:-1]:
            X=layer(X)+X # skip connections
        return self.layers[-1](X)