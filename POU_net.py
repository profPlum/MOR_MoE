import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
from lightning_utils import *
import MOR_Operator

# Verified to work: 7/18/24
# Double verified to work (and reproduce specific partition)
# Triple verified to work (with higher dimensionalities/n_inputs)
class FieldGatingNet(BasicLightningRegressor):
    """
    Essentially a Conv MLP that outputs class probabilities across the field.
    Currently it is not actually a function of the field, just it's size (& the corresponding positions).
    """
    def __init__(self, n_inputs, n_experts, ndims, **kwd_args):
        super().__init__()
        #self.k = k # for top-k selection
        self.ndims=ndims
        self._gating_net = MOR_Operator.MOR_Operator(n_inputs+ndims, n_experts, ndims=ndims, **kwd_args)
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
        #topk = torch.topk(gating_logits, self.k, dim=1).indices
        #gating_logits = gating_logits[:,topk] # first dim is batch_dim
        gating_weights = F.softmax(gating_logits, dim=1)
        return gating_weights#, topk

class POU_net(BasicLightningRegressor):
    ''' POU_net minus the useless L2 regularization '''
    def __init__(self, n_inputs, n_outputs, n_experts=3, ndims=2, lr=0.001, T_max=10,
                 make_expert=MOR_Operator.MOR_Operator, make_gating_net: type=FieldGatingNet, **kwd_args):
        super().__init__()
        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self.gating_net=make_gating_net(n_inputs, n_experts+1, ndims=ndims, **kwd_args) # supports n_inputs!=2
        self.experts=nn.ModuleList([make_expert(n_inputs, n_outputs, ndims=ndims, **kwd_args) for i in range(n_experts)])
        self.r2_score = torchmetrics.R2Score(num_outputs=n_outputs)
        self.explained_variance = torchmetrics.ExplainedVariance()
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
        self.r2_score(y_pred, y)
        self.explained_variance(y_pred, y)
        self.log(f'{val*"val_"}R^2', self.r2_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{val*"val_"}explained_variance', self.explained_variance, on_step=True, on_epoch=True)

    # Verified to work 7/19/24
    def forward(self, X):
        X = torch.as_tensor(X).to(self.device)
        gating_weights = self.gating_net(X)
        prediction = 0
        for i, expert in enumerate(self.experts):
            prediction = prediction + gating_weights[:,i:i+1]*expert(X)
        return prediction