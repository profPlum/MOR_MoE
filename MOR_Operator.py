import torch
import torch.nn as nn
import torch.fft
from lightning_utils import *
import numpy as np

class MOR_Layer(BasicLightningRegressor):
    """ A single Nd MOR operator layer. """
    def __init__(self, in_channels=1, out_channels=1, k_modes=32, ndims=2, **kwd_args):
        super().__init__()
        vars(self).update(locals()); del self.self
        #g_shape = [in_channels, out_channels]+[k_modes]*(ndims-1) + [k_modes//2+1, 2]
        g_shape = [in_channels, out_channels]+[k_modes]*ndims + [2]
        self.g_mode_params = nn.Parameter(torch.randn(*g_shape))
        self.h_mlp=CNN(in_channels,in_channels,k_size=1, ndims=ndims, **kwd_args)
        # Define the weights in the Fourier domain (complex values)

    def forward(self, u):
        u = torch.as_tensor(u).to(self.device)
        assert len(u.shape)==2+self.ndims # +1 for batch, +1 for channels

        # Apply point-wise MLP nonlinearity h(u)
        h_u = self.h_mlp(u) # keeps channel size the same!

        fft_dims = list(range(-self.ndims, 0))
        # Apply Fourier transform (last 2 dims need to be x&y!)

        # should FFT the last self.ndims modes
        u_fft = torch.fft.fftn(h_u, dim=fft_dims)
        u_fft = torch.fft.fftshift(u_fft, dim=fft_dims)

        # Convert to complex dtype & pad the g_mode_params (this will drop extra modes)
        g = torch.view_as_complex(self.g_mode_params)
        g_padded_shape = list(g.shape[:-self.ndims])+list(u_fft.shape[-self.ndims:]) # fuse shapes
        g_padded = torch.zeros(*g_padded_shape, dtype=g.dtype, device=g.device)

        # get total modes to drop for each dimension
        modes_to_drop = (np.asarray(u.shape[-self.ndims:]) - np.asarray(g.shape[-self.ndims:]))

        # 1st part selects input & output channel dimensions, 2nd part inserts g in the center (at the low_freqs)
        low_pass_slices = [slice(None)]*2 + [slice(s_i//2+s_i%2, -s_i//2) for s_i in modes_to_drop]
        g_padded[low_pass_slices] = g
        g_padded = g_padded[None] # add batch dimension

        # Apply learned weights in the Fourier domain
        # (einsum does channel reduction)
        u_fft = torch.einsum('bi...,bio...->bo...', u_fft, g_padded)
        # LEGEND: [b,i,o]:=[batch,in,out] (dimensions)

        # Apply inverse Fourier transform
        u_fft = torch.fft.ifftshift(u_fft, dim=fft_dims)
        u_ifft = torch.fft.ifftn(u_fft, dims=fft_dims)
        # should IFFT the last self.ndims modes

        return u_ifft.real

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