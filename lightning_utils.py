import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics

# Verified to work 7/19/24
class BasicLightningRegressor(L.LightningModule):
    """ Mixin for debugging sub-modules by training them independently. """
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    def training_step(self, batch, batch_idx=None, val=False):
        X, y = batch
        y_pred = self.forward(X).reshape(y.shape)
        loss = F.mse_loss(y_pred, y)
        self.log(f'{val*"val_"}loss', loss.item())
        self.log_metrics(y_pred, y, val) # log additional metrics
        return loss
    def validation_step(self, batch, batch_idx=None):
        return BasicLightningRegressor.training_step(self, batch, batch_idx, val=True)
    def log_metrics(self, y_pred, y, val=False): # override for more metrics
        if not val: self.log_lr()
        #return loss
    def log(self, *args, sync_dist=True, **kwd_args):
        super().log(*args, sync_dist=sync_dist, **kwd_args)
    def log_lr(self):
        scheduler = self.lr_schedulers()
        lrs = scheduler.get_last_lr()
        if type(lrs) in [list, tuple]:
            lrs=sum(lrs)/len(lrs) # simplify
        self.log('lr', lrs, prog_bar=True)

# useful for debugging when vmap won't work
dump_vmap = lambda func: lambda X: torch.stack([func(x) for x in X])

# Verified to work 7/19/24
#class LightningSequential(nn.Sequential, BasicLightningRegressor): pass
class LightningSequential(BasicLightningRegressor):
    def __init__(self, *layers):
        super().__init__()
        self.layers=nn.ModuleList(layers)
    def forward(self, x):
        # Explicitly only call modules in the layers module list
        for module in self.layers:
            x = module(x)
        return x
def CNN(in_size=1, out_size=1, k_size=1, ndims=2,
        n_layers=4, filters=32, activation=nn.SiLU):
    assert n_layers>=1
    assert ndims in [1,2,3]
    ConvLayer = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndims-1]

    # automatically use settings & apply activation
    CNN_layer = lambda in_size, out_size, activation=activation: \
        nn.Sequential(ConvLayer(in_size, out_size, k_size, padding='same'), activation())

    if n_layers==1: # special case, just 1 linear "projection" layer
        layers = [CNN_layer(in_size,out_size,nn.Identity)]
    else:
        layers = [CNN_layer(in_size,filters)] + \
                 [CNN_layer(filters,filters) for i in range(n_layers-2)] + \
                 [CNN_layer(filters,out_size, nn.Identity)]

    return LightningSequential(*layers)
