import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
import numpy as np

# Verified to work 7/19/24
class BasicLightningRegressor(L.LightningModule):
    """ Mixin for debugging sub-modules by training them independently. """
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    def training_step(self, batch, batch_idx=None, val=False):
        X, y = batch
        y_pred = self(X).reshape(y.shape)
        loss = F.mse_loss(y_pred, y)
        self.log(f'{val*"val_"}loss', loss.item(), sync_dist=val, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx=None):
        loss = self.training_step(batch, batch_idx, val=True)
        return loss

# useful for debugging when vmap won't work
dumb_vmap = lambda func: lambda X: torch.stack([func(x) for x in X])

# Verified to work 7/19/24
#class LightningSequential(nn.Sequential, BasicLightningRegressor): pass
class LightningSequential(BasicLightningRegressor):
    def __init__(self, *layers, skip_connections=False):
        super().__init__()
        self.layers=nn.ModuleList(layers)
        self.skip_connections = skip_connections
    def forward(self, X):
        X = self.layers[0](X)
        for layer in self.layers[1:-1]:
            if self.skip_connections: X=(layer(X)+X)/2
            else: X=layer(X)
        return self.layers[-1](X)

# GroupNorm but you can specify num_groups=0 to disable it
ToggleableGroupNorm = lambda num_groups, in_channels: (nn.GroupNorm(num_groups, in_channels) if num_groups>0 else nn.Identity())

def CNN(in_size=1, out_size=1, k_size=1, ndims=2, n_layers=4, hidden_channels=32, activation=nn.SiLU,
        skip_connections=False, hidden_norm_groups=1, out_norm_groups=0, output_activation=False, input_activation=False):
    ''' set hidden_norm_groups=0 or out_norm_groups=0 to disable the group norm there '''
    assert n_layers>=1
    assert ndims in [1,2,3]
    ConvLayer = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndims-1]

    # automatically use settings & apply activation
    CNN_layer = lambda in_size, out_size, activation=activation, use_norm_layer=skip_connections: \
        nn.Sequential(*([ToggleableGroupNorm(hidden_norm_groups, in_size)]*use_norm_layer+[activation(), ConvLayer(in_size, out_size, k_size, padding='same')]))

    input_activation=activation if input_activation else nn.Identity # choose input activation
    if n_layers==1: # special case, just 1 linear "projection" layer
        layers = [CNN_layer(in_size, out_size, input_activation, use_norm_layer=False)]
    else:
        layers = [CNN_layer(in_size,hidden_channels,input_activation, use_norm_layer=False)] + \
                 [CNN_layer(hidden_channels,hidden_channels) for i in range(n_layers-2)] + \
                 [CNN_layer(hidden_channels,out_size)]
    if output_activation: layers[-1].append(activation()) # add output activation
    layers[-1].append(ToggleableGroupNorm(out_norm_groups, out_size))
    model = LightningSequential(*layers, skip_connections=skip_connections)
    return model
