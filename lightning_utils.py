import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
import torch.utils.checkpoint as checkpoint
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

def CNN(in_size=1, out_size=1, k_size=1, ndims=2,
        n_layers=4, hidden_channels=32, activation=nn.SiLU,
        skip_connections=False, scale_weights=False, output_activation=False):
    assert n_layers>=1
    assert ndims in [1,2,3]
    ConvLayer = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndims-1]
    BatchNormLayer = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndims-1]

    output_activation=activation if output_activation else nn.Identity

    # automatically use settings & apply activation
    CNN_layer = lambda in_size, out_size, activation=activation, batch_norm=skip_connections: \
        nn.Sequential(*[BatchNormLayer(in_size)]*batch_norm+[ConvLayer(in_size, out_size, k_size, padding='same'), activation()])

    if n_layers==1: # special case, just 1 linear "projection" layer
        layers = [CNN_layer(in_size,out_size,output_activation, batch_norm=False)]
    else:
        layers = [CNN_layer(in_size,hidden_channels, batch_norm=False)] + \
                 [CNN_layer(hidden_channels,hidden_channels) for i in range(n_layers-2)] + \
                 [CNN_layer(hidden_channels,out_size, output_activation)]

    @torch.no_grad()
    def scale_weights(m):
        scale=5e-3 # same scaling constant found in FNO
        for p in m.parameters(recurse=False):
            p *= scale
    if scale_weights: layers[-1].apply(scale_weights)
    model = LightningSequential(*layers, skip_connections=skip_connections)
    # if scale_weights: model.apply(scale_weights)
    return model
