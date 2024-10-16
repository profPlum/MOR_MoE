import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as L
from lightning_utils import *
import MOR_Operator
import warnings
import random
from contextlib import contextmanager


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
        #self._gating_net = MOR_Operator.MOR_Operator(n_inputs+ndims, n_experts+1, ndims=ndims, **kwd_args)
        self._gating_net = CNN(ndims, n_experts+1, 1, ndims=ndims)#, **kwd_args)
        self._cache_forward=False # whether we should cache the forward call's outputs
        self._cached_forward_results=None # the cached forward call's outputs
        self._cached_forward_shape=None # for sanity check

        # positional encoding cache vars
        self._cached_mesh_shape = None # shape of the X for the cache
        self._cached_mesh_grid = None # cached mesh grid (aka positional encodings)

    def _make_positional_encodings(self, shape):
        # Create coordinate grids using torch.meshgrid
        if tuple(shape) == self._cached_mesh_shape:
            return self._cached_mesh_grid
        assert len(shape)==self.ndims
        coords = [torch.linspace(0,1,steps=dim) for dim in shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        pos_encodings = torch.stack(mesh)[None].to(self.device) # [None] adds batch dim
        self._cached_mesh_shape = tuple(shape)
        self._cached_mesh_grid = pos_encodings
        return pos_encodings

    def forward(self, X):
        # this cache assumes the gating network takes no input (which currently it doesn't)
        if tuple(X.shape)==self._cached_forward_shape:
            assert self._cached_forward_results is not None
            return self._cached_forward_results

        pos_encodings = self._make_positional_encodings(X.shape[-self.ndims:])
        pos_encodings = pos_encodings.expand(X.shape[0], *pos_encodings.shape[1:])
        #X = torch.cat([X, pos_encodings], dim=1)
        gating_logits = self._gating_net(pos_encodings)

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

        # return results
        results = gating_weights, global_topk
        if self._cache_forward:
            self._cached_forward_results=results
            self._cached_forward_shape=tuple(X.shape)
        return results

    @contextmanager
    def cached_gating_weights(self):
        try:
            self._cache_forward=True # tell forward to cache
            yield # yeild nothing during with statement
        finally:
            self._cache_forward=False
            self._cached_forward_results=None # reset cache
            self._cached_forward_shape=None # reset cache

class POU_net(BasicLightningRegressor):
    ''' POU_net minus the useless L2 regularization '''
    def __init__(self, n_inputs, n_outputs, n_experts=5, ndims=2, lr=0.001, momentum=0.9, T_max=10,
                 one_cycle=False, three_phase=False, RLoP=False, RLoP_factor=0.9, RLoP_patience=25,
                 make_optim: type=torch.optim.Adam, make_expert: type=MOR_Operator.MOR_Operator,
                 make_gating_net: type=FieldGatingNet, **kwd_args):
        assert not (one_cycle and RLoP), 'These learning rate schedules are mututally exclusive!'
        super().__init__()
        RLoP_patience = int(RLoP_patience) # cast
        self.save_hyperparameters(ignore=['n_inputs', 'n_outputs', 'ndims', 'simulator', 'make_expert', 'make_gating_net'])

        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self.gating_net=make_gating_net(n_inputs, n_experts, ndims=ndims, **kwd_args) # supports n_inputs!=2
        self.experts=nn.ModuleList([make_expert(n_inputs, n_outputs, ndims=ndims, **kwd_args) for i in range(n_experts)])

        class MetricsModule(L.LightningModule):
            def __init__(self): # these metrics need to be seperated for validatin & training!
                super().__init__()
                self.r2_score = torchmetrics.R2Score(num_outputs=n_outputs)
                self.explained_variance = torchmetrics.ExplainedVariance()
                self.wMAPE = torchmetrics.WeightedMeanAbsolutePercentageError()
                self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()
        self.train_metrics = MetricsModule()
        self.val_metrics = MetricsModule()
        vars(self).update(locals()); del self.self

    def configure_optimizers(self):
        optim_kwd_args = {'lr': self.lr}
        if self.make_optim==torch.optim.SGD:
            optim_kwd_args.update({'momentum': self.momentum, 'nesterov': True})
        optim = self.make_optim(self.parameters(), **optim_kwd_args)

        print('estimated total steps: ', self.trainer.estimated_stepping_batches)
        schedule = {'scheduler': lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=self.T_max, T_mult=2),
                    'interval': 'epoch', 'monitor': 'loss'}
        if self.RLoP: schedule['scheduler'] = lr_scheduler.ReduceLROnPlateau(optim, factor=self.RLoP_factor,
                                                                             patience=self.RLoP_patience)
        elif self.one_cycle:
            schedule['scheduler'] = lr_scheduler.OneCycleLR(optim, max_lr=self.lr, three_phase=self.three_phase,
                                                            total_steps=self.trainer.estimated_stepping_batches)
            schedule['interval'] = 'step'
        return [optim], [schedule]

    def on_before_optimizer_step(self, optimizer):
        from pytorch_lightning.utilities import grad_norm
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms_inf = grad_norm(self, norm_type='inf')
        norms_2 = grad_norm(self, norm_type=2)
        self.log('grad_inf_norm_total', norms_inf['grad_inf_norm_total'].item(), sync_dist=True, reduce_fx='max')
        self.log('grad_2.0_norm_total', norms_2['grad_2.0_norm_total'].item(), sync_dist=True, reduce_fx='mean')

    def log_metrics(self, y_pred, y, val=False):
        super().log_metrics(y_pred, y, val)

        # to_table flattens all dims except for the channel dim (making it tabular)
        to_table = lambda x: x.swapaxes(1, -1).reshape(-1, self.n_outputs)
        y_pred, y = to_table(y_pred), to_table(y)
        metrics = self.val_metrics if val else self.train_metrics

        # simple helper does everything needed to log one metric!
        def log_metric(name, metric=None, on_step=val, on_epoch=True, **kwd_args):
            if metric is None: metric = getattr(metrics, name)
            if on_step: metric(y_pred, y) # update metric
            else: metric.update(y_pred, y)
            self.log(f'{val*"val_"}{name}', metric, on_step=on_step,
                     on_epoch=on_epoch, logger=True, **kwd_args) # log metric

        # we specify the metric itself for the first one to enable a different metric name
        log_metric('R^2', metrics.r2_score, prog_bar=True)
        log_metric('explained_variance')
        log_metric('wMAPE')
        log_metric('sMAPE')

    # Verified to work 7/19/24
    def forward(self, X):
        X = torch.as_tensor(X).to(self.device)
        gating_weights, topk = self.gating_net(X)
        prediction = 0
        for i, k_i in enumerate(topk):
            prediction = prediction + gating_weights[:,i:i+1]*self.experts[k_i](X)
        return prediction
