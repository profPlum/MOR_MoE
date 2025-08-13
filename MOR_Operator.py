import torch
import torch.nn as nn
import torch.fft
from lightning_utils import *
import numpy as np
import warnings

# Verified to work: 9/2/14
def make_rfft_corner_slices(img1_shape, img2_shape, fft_dims=None, rfft=True, verbose=True):
    ''' Creates slices of low-mode corners that match both img1_shape and img2_shape. '''
    import itertools
    if (np.asarray(img1_shape)<np.asarray(img2_shape)).any():
        warnings.warn('you are downsampling the parameters!')
    min_shape = np.minimum(img1_shape, img2_shape) # find shape compatible with both
    if fft_dims is None: fft_dims = np.arange(len(min_shape)) # None implies everything
    else: fft_dims = np.arange(len(min_shape))[fft_dims] # standardize it
    if rfft: min_shape[fft_dims[-1]] *= 2 # last dim is only 1/2! (also last is last in FFT dims)
    if verbose: print(f'{min_shape=}')
    valid_corner_slices = []
    for i, s_i in enumerate(min_shape):
        if i in fft_dims:
            # GOTCHA: parens in -(s_i//2) is needed to avoid surprising behavior with floor + negatives
            # Also we round up b/c fft puts 0 mode on the positive side so it can have extra element.
            valid_corner_slices.append([slice(None, s_i//2+s_i%2), slice(-(s_i//2), None)])
        else: valid_corner_slices.append([slice(None)]) # else select entire dim
    if rfft: del valid_corner_slices[fft_dims[-1]][-1] # last dim has only positive freqs (due to symmetry)
    if verbose: print(f'{valid_corner_slices=}')
    return itertools.product(*valid_corner_slices) # cartesian product gives all corners

class MOR_Layer(BasicLightningRegressor):
    """ A single Nd MOR operator layer. """
    def __init__(self, in_channels=1, out_channels=1, k_modes=32, ndims=2, batch_norm=True, **kwd_args):
        super().__init__()
        vars(self).update(locals()); del self.self

        # why is this a function?
        if type(k_modes) is int:
                k_modes = ndims*[k_modes]
        def make_g(in_channels, out_channels):
            g_shape = [in_channels, out_channels]+k_modes[:ndims-1] + [k_modes[-1]//2+1, 2]
            scale = 5e-3 #1.0/(in_channels**0.5)#out_channels) # similar to kaiming initializaiton
            param = torch.randn(*g_shape)*scale
            return nn.Parameter(param)

        # Define the weights in the Fourier domain (complex values)
        mlp_channels = [in_channels]*2
        g_channels = [in_channels, out_channels]
        #if mlp_second: mlp_channels, g_channels = g_channels, mlp_channels

        BatchNormLayer = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndims-1]
        if batch_norm: self.batch_norm_layer = BatchNormLayer(in_channels)

        self.g_mode_params = make_g(*g_channels)
        self.h_mlp=CNN(*mlp_channels, k_size=1, n_layers=2, ndims=ndims, output_activation=True, **kwd_args)

    def forward(self, u):
        u = torch.as_tensor(u, device=self.device)
        assert len(u.shape)==2+self.ndims # +1 for batch, +1 for channels
        # u.shape==[batch, in_channels, x, y, ...]

        # use the batch norm layer if it's here
        try: u = self.batch_norm_layer(u)
        except AttributeError: pass

        # Apply point-wise MLP nonlinearity h(u)
        u = self.h_mlp(u)

        # Apply Fourier transform (last ndims need to be spatial!)
        # should FFT the last self.ndims modes, also we pad (if needed) to make low-pass work
        fft_dims = list(range(-self.ndims, 0))
        u_fft = torch.fft.rfftn(u, dim=fft_dims)

        g = torch.view_as_complex(self.g_mode_params) # Convert to complex dtype (makes shape "correct")
        g_padded_shape = list(g.shape[:-self.ndims])+list(u_fft.shape[-self.ndims:]) # fuse shapes

        # insert G into "G_padded" s.t. it applies a low pass filter
        # & Apply learned weights in the Fourier domain (einsum does channel reduction)
        if tuple(g.shape)==tuple(g_padded_shape):
            g_padded=g # special optimization to juice out an extra time-step: just one einsum
            g_padded = g_padded[None] # add batch dimension
            assert u_fft.shape[1]==g_padded.shape[1]==g.shape[0] # check channel compatibility
            u_fft = torch.einsum('bi...,bio...->bo...', u_fft, g_padded) # LEGEND: [b,i,o]:=[batch,in,out] (dimensions)
        else: # really we just insert the einsum piece-wise into u_fft_out (padded), since it's more efficient
            warnings.warn('Not using special optimization! tuple(g.shape)!=tuple(g_padded_shape)')
            # NOTE: einsum only changes the number of channels! [batch, g_out_channels, *spatial_dims]
            u_fft_out = torch.zeros(u_fft.shape[0], g.shape[1], *u_fft.shape[2:], dtype=u_fft.dtype, device=u_fft.device)
            assert u_fft.shape[1]==g.shape[0]
            for corner in make_rfft_corner_slices(g_padded_shape, g.shape, fft_dims=fft_dims, verbose=False):
                u_fft_out[corner] = torch.einsum('bi...,bio...->bo...', u_fft[corner], g[corner][None]) #[None] adds batch dimension to g
                # LEGEND: [b,i,o]:=[batch,in,out] (dimensions)
            u_fft = u_fft_out

        # Apply inverse Fourier transform
        u_ifft = torch.fft.irfftn(u_fft, s=u.shape[-self.ndims:], dim=fft_dims).real
        # should IFFT the last self.ndims modes, also crop/pad to original shape

        assert u_ifft.shape[-self.ndims:]==u.shape[-self.ndims:]

        return u_ifft

class MOR_Operator(BasicLightningRegressor):
    """
    Essentially a stack of MORLayer-Nd layers + skip connections.
    Without skip-connections this operator doesn't work at all
    (assuming multiple layers), b/c of 1 channel bottleneck & b/c
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, n_layers=4, **kwd_args):
        super().__init__()
        kwd_args['hidden_channels'] = hidden_channels # make h(x) hidden_channels=MOR_Operator.hidden_channels
        #self.layers = nn.ModuleList([MOR_Layer(in_channels, hidden_channels, batch_norm=False, **kwd_args)] +
        #    [MOR_Layer(hidden_channels, hidden_channels, batch_norm=True, **kwd_args) for i in range(n_layers-2)]+
        #    [MOR_Layer(hidden_channels, out_channels, batch_norm=True, **kwd_args)])

        ndims = {'ndims': kwd_args['ndims']} if 'ndims' in kwd_args else {}
        ProjLayer = lambda *args, **kwd_args: CNN(*args, n_layers=2, **ndims, **kwd_args)
        self.layers = nn.ModuleList([MOR_Layer(in_channels, hidden_channels, batch_norm=False, **kwd_args)] + #[ProjLayer(in_channels, hidden_channels)] +
            [MOR_Layer(hidden_channels, hidden_channels, batch_norm=True, input_activation=True, **kwd_args) for i in range(n_layers-1)]+
            [ProjLayer(hidden_channels, out_channels, scale_weights=True, input_activation=True)]) # this is like having an extra h(x) at the end
    def forward(self, X):
        X = self.layers[0](X)
        for layer in self.layers[1:-1]:
            X=(layer(X)+X)/2 # skip connections
        return self.layers[-1](X)
