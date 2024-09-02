import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
from lightning_utils import *
import MOR_Operator
import warnings
import random

# Verified to work: 7/18/24
# Double verified to work (and reproduce specific partition)
# Triple verified to work (with higher dimensionalities/n_inputs)
class FieldGatingNet(BasicLightningRegressor):
    """
    Essentially a Gating Operator that outputs class probabilities across the field.
    It is now a function of the field itself and the coordinates of the field positions!
    Also it will implicitly add the option of a 'null expert' (an expert that just predicts 0).
    And it adds some small amount of noise to the gating logits to encourage exploration.
    """
    def __init__(self, n_inputs, n_experts, ndims, k=2, noise_sd=0.005, **kwd_args):
        super().__init__()
        self.k = k # for (global) top-k selection
        if k<2: warnings.warn('K<2 means the gating network might not learn to gate properly.')
        self.ndims = ndims
        self.noise_sd = noise_sd

        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self._gating_net = MOR_Operator.MOR_Operator(n_inputs+ndims, n_experts+1, ndims=ndims, **kwd_args)
    def _make_positional_encodings(self, shape):
        # Create coordinate grids using torch.meshgrid
        assert len(shape)==self.ndims
        coords = [torch.linspace(0,1,steps=dim) for dim in shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        pos_encodings = torch.stack(mesh)[None] # [None] adds batch dim
        return pos_encodings.to(self.device)
    def forward(self, X):
        pos_encodings = self._make_positional_encodings(X.shape[-self.ndims:])
        pos_encodings = pos_encodings.expand(X.shape[0], *pos_encodings.shape[1:])
        X = torch.cat([X, pos_encodings], dim=1)
        gating_logits = self._gating_net(X)

        # global average pooling to identify top-k global experts (across spatial & BATCH dims!)
        global_logits = torch.mean(gating_logits, dim=[0]+list(range(-self.ndims,0)))
        if self.training:
            global_logits = global_logits + torch.randn_like(global_logits, requires_grad=False)*self.noise_sd
        assert len(global_logits.shape)==1 # 1D

        if random.random() < 0.01: # keep an eye on this it seems suspiciously low...
            global_logit_sd = (gating_logits.var(dim=[0]+list(range(-self.ndims,0))).mean()**0.5).item()
            print(f'{global_logit_sd=}')

        # add in the obligitory null expert (always the last index in the softmax)
        global_topk = torch.topk(global_logits[:-1], self.k, dim=0).indices # we don't want to select null expert twice!
        global_topk = torch.cat([global_topk, torch.tensor([-1], device=global_topk.device, dtype=global_topk.dtype)])
        gating_logits = gating_logits[:, global_topk] # first dim is batch_dim
        gating_weights = F.softmax(gating_logits, dim=1)[:,:-1]
        global_topk = global_topk[:-1]
        # after the 'null expert' has influenced the softmax normalization
        # it can disappear (we won't waste any flops on it...)

        return gating_weights, global_topk

class POU_net(BasicLightningRegressor):
    ''' POU_net minus the useless L2 regularization '''
    def __init__(self, n_inputs, n_outputs, n_experts=5, ndims=2, lr=0.001, T_max=10,
                 make_expert=MOR_Operator.MOR_Operator, make_gating_net: type=FieldGatingNet, **kwd_args):
        super().__init__()
        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self.gating_net=make_gating_net(n_inputs, n_experts, ndims=ndims, **kwd_args) # supports n_inputs!=2
        self.experts=nn.ModuleList([make_expert(n_inputs, n_outputs, ndims=ndims, **kwd_args) for i in range(n_experts)])

        class MetricsModule(L.LightningModule):
            def __init__(self): # these metrics need to be seperated for validatin & training!
                super().__init__()
                self.r2_score = torchmetrics.R2Score(num_outputs=n_outputs)
                self.explained_variance = torchmetrics.ExplainedVariance()
        self.train_metrics = MetricsModule()
        self.val_metrics = MetricsModule()
        vars(self).update(locals()); del self.self

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=self.T_max, T_mult=2)
        #lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.T_max)
        return [optim], [lr_schedule]

    def log_metrics(self, y_pred, y, val=False):
        super().log_metrics(y_pred, y, val)

        # to_table flattens all dims except for the channel dim (making it tabular)
        to_table = lambda x: x.swapaxes(1, -1).reshape(-1, self.n_outputs)
        y_pred, y = to_table(y_pred), to_table(y)
        metrics = self.val_metrics if val else self.train_metrics
        metrics.r2_score(y_pred, y)
        metrics.explained_variance(y_pred, y)
        self.log(f'{val*"val_"}R^2', metrics.r2_score, on_step=not val, on_epoch=True, prog_bar=True)
        self.log(f'{val*"val_"}explained_variance', metrics.explained_variance, on_step=True, on_epoch=True)

    # Verified to work 7/19/24
    def forward(self, X):
        X = torch.as_tensor(X).to(self.device)
        gating_weights, topk = self.gating_net(X)
        prediction = 0
        for i, k_i in enumerate(topk):
            prediction = prediction + gating_weights[:,i:i+1]*self.experts[k_i](X)
        return prediction
