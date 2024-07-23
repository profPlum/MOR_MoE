import torch
import torch.nn as nn
import torch.fft
from lightning_utils import BasicLightningRegressor, CNN2d

# Verified to work 7/19/24
class MORLayer2D(BasicLightningRegressor):
    """ A single 2D MOR operator layer. """
    def __init__(self, k_modes=32, activation=nn.SiLU):
        super().__init__()
        vars(self).update(locals()); del self.self
        self.g_mode_params = nn.Parameter(torch.randn(k_modes, k_modes//2+1, 2))
        self.h_mlp=CNN2d(k_size=1, activation=activation)
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

class MOR2dOperator(BasicLightningRegressor):
    """
    Essentially a stack of MORLayer2D layers + skip connections.
    Without skip-connections this operator doesn't work at all
    (assuming multiple layers), b/c of 1 channel bottleneck.
    """
    def __init__(self, n_layers=4, **kwd_args):
        super().__init__()
        self.layers = nn.ModuleList([MORLayer2D(**kwd_args) for i in range(n_layers)])
    def forward(self, X):
        for layer in self.layers[:-1]:
            X=layer(X)+X
        return self.layers[-1](X)