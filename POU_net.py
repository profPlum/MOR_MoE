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

import utils

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
    def __init__(self, n_inputs, n_experts, ndims, k=2, trig_encodings=True, noise_sd=0.0):
        super().__init__()
        self.k = min(k, n_experts) # for (global) top-k selection
        if k<2 and n_experts>=2: warnings.warn('K<2 means the gating network might not learn to gate properly.')
        self.ndims = ndims
        self.noise_sd = noise_sd
        self._trig_encodings = trig_encodings

        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self._gating_net = CNN(ndims*(1+trig_encodings), n_experts+1, 1, ndims=ndims)#, **kwd_args)
        self._cache_forward=False # whether we should cache the forward call's outputs
        self._cached_forward_results=None # the cached forward call's outputs
        self._cached_forward_shape=None # for sanity check

        # positional encoding cache vars
        self._cached_mesh_shape = None # shape of the X for the cache
        self._cached_mesh_grid = None # cached mesh grid (aka positional encodings)

    def _make_positional_encodings(self, shape):
        # Create coordinate grids using torch.meshgrid
        if tuple(shape) == self._cached_mesh_shape and self._cached_mesh_grid.device==self.device:
            return self._cached_mesh_grid
        assert len(shape)==self.ndims
        linspace = lambda dim: torch.linspace(0,1,steps=dim)
        if self._trig_encodings:
            linspace = lambda dim: torch.linspace(0,1,steps=dim+1)[:-1]*2*np.pi
        coords = [linspace(dim) for dim in shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        if self._trig_encodings:
            mesh = [torch.cos(x) for x in mesh] + [torch.sin(x) for x in mesh]
        pos_encodings = torch.stack(mesh)[None].to(self.device) # [None] adds batch dim
        self._cached_mesh_shape = tuple(shape)
        self._cached_mesh_grid = pos_encodings
        return pos_encodings

    def forward(self, X):
        # this cache assumes the gating network takes no input (which currently it doesn't)
        if tuple(X.shape)==self._cached_forward_shape:
            assert self._cached_forward_results is not None
            return self._cached_forward_results

        with torch.no_grad():
            pos_encodings = self._make_positional_encodings(X.shape[-self.ndims:])
        #pos_encodings = pos_encodings.expand(X.shape[0], *pos_encodings.shape[1:])
        gating_logits = self._gating_net(pos_encodings)
        if self.training and self.noise_sd>0: gating_logits = gating_logits + torch.randn_like(gating_logits, requires_grad=False)*self.noise_sd
        global_logits = torch.randn(gating_logits.shape[1], requires_grad=False) # random selection
        assert len(global_logits.shape)==1 # 1D

        # add in the obligitory null expert (always the last index in the softmax)
        global_topk = torch.topk(global_logits[:-1], self.k, dim=0, sorted=False).indices # we don't want to select null expert twice!
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

# these metrics need to be seperated for validation & training!
class MetricsModule(L.LightningModule):
    def __init__(self, parent_module:L.LightningModule, n_outputs:int, prefix=''):
        super().__init__()

        # this list-trick prevents parent module being registered as a sub-module!
        self._parent_module = [parent_module]
        self.n_outputs=n_outputs
        self.prefix=prefix

        self.r2_score = torchmetrics.R2Score(num_outputs=n_outputs)
        self.explained_variance = torchmetrics.ExplainedVariance()
        self.wMAPE = torchmetrics.WeightedMeanAbsolutePercentageError()
        self.sMAPE = torchmetrics.SymmetricMeanAbsolutePercentageError()

    def log_metrics(self, y_pred, y):
        with torch.inference_mode():
            # to_table flattens all dims except for the channel dim (making it tabular)
            to_table = lambda x: x.swapaxes(1, -1).reshape(-1, self.n_outputs)
            y_pred, y = to_table(y_pred), to_table(y)

            # simple helper does everything needed to log one metric!
            def log_metric(name, metric=None, on_step=False, on_epoch=True, **kwd_args):
                if metric is None: metric = getattr(self, name)
                if on_step: metric(y_pred, y) # update metric
                else: metric.update(y_pred, y)
                self._parent_module[0].log(f'{self.prefix}{name}', metric, on_step=on_step,
                                           on_epoch=on_epoch, logger=True, **kwd_args) # log metric

            # we specify the metric itself for the first one to enable a different metric name
            log_metric('R^2', self.r2_score, prog_bar=not self.prefix)
            log_metric('explained_variance')
            log_metric('wMAPE')
            log_metric('sMAPE')

''' # we did this implicitly instead of using the class
class ZeroExpert(L.LightningModule):
    def __init__(self, sigma=False):
        if sigma: self._sigma=nn.Parameter(torch.randn([]))
    def forward(self, *args):
        try: return 0, F.softplus(self._sigma)
        except AttributeError: return 0
'''

class POU_net(L.LightningModule):
    ''' POU_net minus the useless L2 regularization '''
    def __init__(self, n_inputs, n_outputs, n_experts=3, ndims=2, lr=0.001, momentum=0.9, T_max=10,
                 one_cycle=False, three_phase=False, RLoP=False, RLoP_factor=0.9, RLoP_patience:int=25,
                 make_optim: type=torch.optim.Adam, make_expert: type=MOR_Operator.MOR_Operator,
                 make_gating_net: type=FieldGatingNet, trig_encodings=True, **kwd_args):
        assert not (one_cycle and RLoP), 'These learning rate schedules are mututally exclusive!'
        super().__init__()
        RLoP_patience = int(RLoP_patience) # cast
        self.save_hyperparameters(ignore=['n_inputs', 'n_outputs', 'ndims', 'simulator', 'make_expert', 'make_gating_net'])

        # NOTE: The gating_net implicitly adds a "ZeroExpert"
        self.gating_net=make_gating_net(n_inputs, n_experts, ndims=ndims, trig_encodings=trig_encodings) # supports n_inputs!=2
        self.experts=nn.ModuleList([make_expert(n_inputs, n_outputs, ndims=ndims, **kwd_args) for i in range(n_experts)])

        self.train_metrics = MetricsModule(self, n_outputs)
        self.val_metrics = MetricsModule(self, n_outputs, prefix='val_')
        vars(self).update(locals()); del self.self; del self.kwd_args

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

    # Verified to work 7/19/24
    def forward(self, X):
        X = torch.as_tensor(X).to(self.device)
        gating_weights, topk = self.gating_net(X)
        prediction = 0
        for i, k_i in enumerate(topk):
            prediction = prediction + gating_weights[:,i:i+1]*self.experts[k_i](X)
        return prediction

    def training_step(self, batch, batch_idx=None, val=False):
        X, y = batch
        y_pred = self(X).reshape(y.shape)
        loss = F.mse_loss(y_pred, y)
        self.log(f'{val*"val_"}loss', loss.item(), sync_dist=val, prog_bar=not val)
        self._log_metrics(y_pred, y, val) # log additional metrics
        return loss

    def validation_step(self, batch, batch_idx=None, data_loader_idx=0):
        loss = self.training_step(batch, batch_idx, val=True)
        if data_loader_idx==0:
            self.log('hp_metric', loss.item(), sync_dist=True)
        return loss

    def _log_metrics(self, y_pred, y, val=False):
        if not val: self._log_lr()
        metrics = self.val_metrics if val else self.train_metrics
        metrics.log_metrics(y_pred, y)

    def _log_lr(self):
        scheduler = self.lr_schedulers()
        lrs = scheduler.get_last_lr()
        if type(lrs) in [list, tuple]:
            lrs=sum(lrs)/len(lrs) # simplify
        self.log('lr', lrs, on_step=True, prog_bar=True)

import model_agnostic_BNN

class PPOU_net(POU_net): # Not really, it's POU+VI
    def __init__(self, n_inputs, n_outputs, train_dataset_size, *args, total_variance=True, prior_cfg={}, **kwd_args):
        # we double output channels to have the sigma predictions too
        super().__init__(n_inputs*2, n_outputs*2, *args, **kwd_args)

        # make VI reparameterize our entire model
        model_agnostic_BNN.model_agnostic_dnn_to_bnn(self, train_dataset_size, prior_cfg=prior_cfg)

        # add additional set of metrics for validating aleatoric UQ itself compared to error
        self.val_UQ_metrics = MetricsModule(self, n_outputs, prefix='val_UQ_')
        self._zero_expert_sigma=nn.Parameter(torch.randn([1]))
        self._total_variance=total_variance

    def forward(self, X, Y=None):
        ''' crazy forward method that does everything needed for total variance of mixture distribution with zero expert '''
        X = torch.as_tensor(X).to(self.device)
        if Y is None: Y = torch.zeros(1,device=X.device, dtype=X.dtype).expand(*X.shape)
        X = torch.cat([X,Y], axis=1)

        gating_weights, topk = self.gating_net(X)
        total_expectation = 0 # E[Y] = E[E[Y|Z]]
        total_variance = 0 # Var[Y] = E[Var[Y|Z]]+Var[E[Y|Z]]
        mus = [] # necessary for 2nd term in total variance eq.

        for i, k_i in enumerate(topk):
            pred_i = self.experts[k_i](X)
            mu = pred_i[:, :pred_i.shape[1]//2]
            sigma = F.softplus(pred_i[:, pred_i.shape[1]//2:]) # enforce positivity
            mus.append(mu)

            total_expectation = total_expectation + gating_weights[:,i:i+1]*mu
            total_variance = total_variance + gating_weights[:,i:i+1]*sigma**2

        if self._total_variance: # add in the explained variance term (2nd term) = Var(mus) = Var[E[Y|Z]]
            total_variance = total_variance + sum([gating_weights[:,i:i+1]*(mu-total_expectation)**2 for i, mu in enumerate(mus)])
        mus.clear() # paranoia related to memory management

        # handle zero expert
        zero_expert_gating_weights = 1-gating_weights.sum(axis=1, keepdim=True) # recover zero expert weights
        total_variance = total_variance + zero_expert_gating_weights*F.softplus(self._zero_expert_sigma)**2 # for 1st term
        if self._total_variance:
            total_variance = total_variance + zero_expert_gating_weights*total_expectation**2 # for 2nd term (total_expectation**2==(0-total_expectation)**2)

        return total_expectation, total_variance**0.5

    '''
    # original forward before probabilistic considerations
    def forward(self, X, Y=None):
        if Y is None: Y = torch.zeros(1,device=X.device, dtype=X.dtype).expand(*X.shape)
        X = torch.cat([X,Y], axis=1)
        pred = super().forward(X)

        mu_pred = pred[:, :pred.shape[1]//2]
        sigma_pred = F.softplus(pred[:, pred.shape[1]//2:])
        return mu_pred, sigma_pred
    '''

    def training_step(self, batch, batch_idx=None, val=False):
        X, y = batch
        y_pred_all = self(X)
        y_pred_mu, y_pred_sigma = y_pred_all

        #num_batches = len(self.trainer.train_dataloader.dataset)//self.trainer.train_dataloader.batch_size # sneakily extract from PL
        kl_loss = self.get_kl_loss()#/(num_batches*y.numel()) # (weighted)
        loss = model_agnostic_BNN.nll_regression(y_pred_mu, y, y_pred_sigma=y_pred_sigma, reduction=torch.mean) + kl_loss # posterior loss

        self.log(f'{val*"val_"}loss', loss.item(), sync_dist=val, prog_bar=not val)
        if not val: self.log('kl_loss', kl_loss.item(), sync_dist=val, prog_bar=True)
        self._log_metrics(y_pred_all, y, val) # log additional metrics (mu & sigma variants)

        return loss

    def _log_metrics(self, y_pred: tuple, y: torch.Tensor, val=False):
        y_pred_mu, y_pred_sigma = y_pred # break apart pred tuple
        super()._log_metrics(y_pred_mu, y, val=val) # log regular mu metrics & lr (implicitly)

        if not val: return # UQ metrics for training would be overkill...
        sigma_to_mad_coef = 0.7978865 # this magic constant can be multiplied with sigma of a 1d guassian to obtain the MAD! E[|X-E[X]|] (expected absolute error)
        with torch.inference_mode():
            y_abs_error=(y-y_pred_mu).abs() # y_pred_mu is to y, as y_pred_sigma is to y_abs_error
            y_pred_MAD = y_pred_sigma*sigma_to_mad_coef
            self.val_UQ_metrics.log_metrics(y_pred_MAD, y_abs_error)
            self.log(f'val_UQ_MAE', (y_abs_error-y_pred_MAD).abs().mean(), sync_dist=True, on_epoch=True, on_step=False)
