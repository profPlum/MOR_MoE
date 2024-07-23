import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

# Verified to work 7/19/24
class BasicLightningRegressor(L.LightningModule):
    """ Mixin for debugging sub-modules by training them independently. """
    def training_step(self, batch, batch_idx=None, val=False):
        X, y = batch
        y_pred = self.forward(X)
        loss = F.mse_loss(y, y_pred.reshape(y.shape))
        self.log(f'{val*"val_"}loss', loss.item(), prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    def validation_step(self, batch):
        return BasicLightningRegressor.training_step(self, batch, val=True)

# TODO: generalize across dimensionality...
# Verified to work 7/19/24
class LightningSequential(nn.Sequential, BasicLightningRegressor): pass
CNN2d_layer = lambda in_size, out_size, k_size, activation=nn.SiLU: \
    nn.Sequential(nn.Conv2d(in_size, out_size, k_size, padding='same'), activation())
CNN2d = lambda in_size=1, out_size=1, k_size=3, activation=nn.SiLU, n_layers=5, filters=32: \
    LightningSequential(*([CNN2d_layer(in_size,filters,k_size,activation)] +
                          [CNN2d_layer(filters,filters,k_size,activation) for i in range(n_layers-2)] +
                          [CNN2d_layer(filters,out_size,k_size,nn.Identity)]))