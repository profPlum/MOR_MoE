import os, random
import torch, numpy as np
from torch import nn
import torch.nn.utils.parametrize as parametrize

# NOTE: We changed it to use sum because the prior p(theta)=p(theta_0)p(theta_1)...p(theta_n) is a product of gaussians
# which means that more parameters will increase the KL loss. Also KL divergence takes a log which implies the sum.
def kl_div(mu_q, sigma_q, mu_p, sigma_p):
    """
    Calculates kl divergence between two gaussians (Q || P)

    Parameters:
         * mu_q: torch.Tensor -> mu parameter of distribution Q
         * sigma_q: torch.Tensor -> sigma parameter of distribution Q
         * mu_p: float -> mu parameter of distribution P
         * sigma_p: float -> sigma parameter of distribution P

    returns torch.Tensor of shape 0
    """
    assert not (torch.is_complex(mu_q) or torch.is_complex(sigma_q))

    mu_p = torch.as_tensor(mu_p)
    sigma_p = torch.as_tensor(sigma_p)
    kl = torch.log(sigma_p) - torch.log(sigma_q) + \
        (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
    return kl.sum()

# NOTE: We changed it to use sum because the prior p(theta)=p(theta_0)p(theta_1)...p(theta_n) is a product of gaussians
# which means that more parameters will increase the KL loss. Also KL divergence is log(prob) which implies the sum.
def get_kl_loss(m):
    kl_loss = 0.0
    for layer in m.modules():
        if hasattr(layer, "kl_loss"):
            kl_loss += layer.kl_loss()
    return kl_loss

_pt_nll_classificiation = torch.nn.NLLLoss(reduction='sum') # sum needed for the true NLL
nll_classification = lambda y_pred, y, **kwd_args: _pt_nll_classificiation(y_pred, y)
nll_regression_basic = lambda y_pred, y, **kwd_args: ((y_pred-y)**2).sum()/2 # Truest NLL (for regression)!

# Truest NLL (for regression)!
nll_sigma_ = lambda y_pred, y, y_pred_sigma: ((y_pred-y)**2).sum()/(2*y_pred_sigma**2) + torch.log(y_pred_sigma)
nll_regression = lambda y_pred, y: nll_sigma(y_pred[:,0], y, y_pred_sigma=y_pred[:,1])
# this assumes that first dim after batch seperates mu, sigma preds

_get_rho = lambda sigma: np.log(np.expm1(sigma)+1e-20)

# This is simpler & MUCH faster version of _BayesianParameterization (previous version was 1.5x slower per epoch!)
# NOTE: I opted not to do RhoParametrization on this Parameterization itself because it seemed crazy & confusing to have
# a Parametrized_BayesianParameterization Parametrization. See what I mean? I can barely say the damn thing.
class _BayesianParameterization(nn.Module):
    def __init__(self, mu_params, posterior_mu_init=None, posterior_sigma_init=0.0486,
                 prior_mu=0.0, prior_sigma=1.0):
        """ For MLE-pretraining: posterior_mu_init and/or prior_mu can be None if you want to copy their values
            from mu_params (for MLE-pretraining). Also if you want to do MOPED-style prior-sigma setting then leave
            prior_sigma=None (with prior_mu=None as well). """
        super().__init__()
        if torch.is_complex(mu_params):
            mu_params = torch.view_as_real(mu_params)

        # prior_sigma is None implies prior_mu is None
        if prior_sigma is None:
            assert prior_mu is None, 'prior_sigma can only be None if prior_mu is also None (which gives MOPED-style prior)'

        # informed prior_mu if not specified
        self.register_buffer('prior_mu', mu_params.clone() if prior_mu is None else torch.Tensor([prior_mu]).to(mu_params.device))
        self.register_buffer('_prior_sigma', torch.Tensor([-1.0 if prior_sigma is None else prior_sigma]).to(mu_params.device))
        # MOPED-style if not specified (_prior_sigma<0.0 is obviously invalid and flags MOPED-style prior_sigma in the property).
        # NOTE: Access prior_sigma via the property self.prior_sigma!
        # TODO: check this MOPED implementation is close enough to the real paper at some point...

        assert 1e-3 < posterior_sigma_init < 0.2, 'High values of posterior_sigma_init cause divergence (and NaNs).'
        posterior_rho_init = _get_rho(posterior_sigma_init)

        with torch.no_grad():
            # don't assign the whole thing homogeneously because we need symmetry breaking!
            if posterior_mu_init is not None: # leave in the option to copy mu from existing model
                mu_params[:] = posterior_mu_init + torch.randn_like(mu_params)*0.1
            posterior_rho_init = _get_rho(posterior_sigma_init) # convert sigma to rho
            # Create param copy for rho/sigma:
            self._rho_params = nn.Parameter(posterior_rho_init + torch.randn_like(mu_params)*0.1)
            #self._rho_params = nn.Parameter(torch.ones_like(mu_params)*posterior_rho_init)
        assert self._rho_params.requires_grad and mu_params.requires_grad

    @property
    def prior_sigma(self): # property enables weight sharing between mu_prior & sigma_prior for MOPED-style informed prior
        return self.prior_mu.abs() if self._prior_sigma<0.0 else self._prior_sigma
        # self._prior_sigma<0.0 implies that we are doing MOPED-style informed prior

    def forward(self, mu_params):
        is_complex = torch.is_complex(mu_params)
        if is_complex:
            mu_params = torch.view_as_real(mu_params)

        # Here we make the seed for *bayesian sampling* unique across processes for better batch parallelism!
        # NOTE: It's ugly b/c: don't want it to make the global seed unique per-process
        # GOTCHA: It will never be reproducible b/c process pids aren't deterministic/controllable!
        pid_seed = random.randint(0, int(2**63)-int(1e9)) + os.getpid()
        def torch_randn_like(input, seed=None): # supports seeding ...unlike torch.randn_like()
            gen = None if seed is None else torch.Generator(device=input.device).manual_seed(seed)
            return torch.randn(input.size(), generator=gen, dtype=input.dtype,
                               layout=input.layout, device=input.device)

        standard_normal = torch_randn_like(self._rho_params, seed=pid_seed)
        sigma_params = nn.functional.softplus(self._rho_params)

        # Update KL loss based on mu_params & sigma_params
        self._kl_loss = kl_div(mu_params, sigma_params, self.prior_mu, self.prior_sigma)
        sampled_values = mu_params+sigma_params*standard_normal
        if is_complex:
            sampled_values = torch.view_as_complex(sampled_values)
        return sampled_values

    def kl_loss(self): # this method apparently is sufficient to work with get_kl_loss(model) as-is!
        return self._kl_loss

# NOTE: this is essentially the new dnn_to_bnn() but also more versatile
def model_agnostic_dnn_to_bnn(dnn: nn.Module, prior_cfg: dict = {}):
    if 'Bayesian' in type(dnn).__name__: return

    # apply _BayesianParameterization
    def visit_parametrize(module: nn.Module):
        #print(module)
        for name, param in list(module.named_parameters(recurse=False)):
            assert '.' not in name
            if param.requires_grad:
                parametrize.register_parametrization(module, name, _BayesianParameterization(param, **prior_cfg))
            else: print('param: ', name, 'doesnt require gradient!', flush=True)
    dnn.apply(visit_parametrize)

    # redefine class to use cached forward operation for speed & consistency
    def forward(self, *args, **kwargs):
        with parametrize.cached():
            return super(type(self), self).forward(*args, **kwargs)
    methods =  {'forward': forward, 'get_kl_loss': get_kl_loss}
    dnn.__class__  = type(f'Bayesian{dnn.__class__.__name__}', (type(dnn),), methods)

# GOTCHA: This is still technically incorrect because NLL loss is supposed to sum across features too...
# TODO: Test! It could work to replace more complicated existing interface...
def model_agnostic_dnn_to_bnn_auto_KL(dnn: nn.Module, dataset_size=None,
                                      prior_weight=1.0, prior_cfg: dict = {}):
    if 'Bayesian' in type(dnn).__name__: return
    if dataset_size is None: # default assumption becomes that model parameters & dataset size are balanced
        dataset_size = len(torch.nn.utils.parameters_to_vector(dnn.parameters()))
        # ...this is equivalent to KL taking the mean across parameters

    # apply _BayesianParameterization
    def visit_parametrize(module: nn.Module):
        #print(module)
        for name, param in list(module.named_parameters(recurse=False)):
            assert '.' not in name
            if param.requires_grad:
                parametrize.register_parametrization(module, name, _BayesianParameterization(param, **prior_cfg))
            else: print('param: ', name, 'doesnt require gradient!', flush=True)
    dnn.apply(visit_parametrize)

    # redefine class to use cached forward operation for speed & consistency
    def forward(self, *args, **kwargs):
        with parametrize.cached():
            output = super(type(self), self).forward(*args, **kwargs)
            if self.training: (prior_weight*self.get_kl_loss()/dataset_size).backward()
        return output
    methods =  {'forward': forward, 'get_kl_loss': get_kl_loss}
    dnn.__class__  = type(f'Bayesian{dnn.__class__.__name__}', (type(dnn),), methods)

# TODO: embed the moment accumulation functions into the forward method of the class
################################ BNN sampling utility functions: ################################

def clear_cache():
    ''' clear pytorch cuda cache '''
    import torch, gc
    while gc.collect(): pass
    torch.cuda.empty_cache()

def get_BNN_pred_distribution(bnn_model, x_input, n_samples=100, no_grad=True):#, cleanup_freq=15):
    '''
    If you just want moments use get_BNN_pred_moments() instead as it is *much* more memory efficient (e.g. for large sample sizes). But this is still useful if you want an actual distribution.
    '''
    if no_grad:
        with torch.no_grad():
            return get_BNN_pred_distribution(bnn_model, x_input, n_samples, no_grad=False)
    preds = []
    for i in range(n_samples):
        preds.append(bnn_model(x_input).cpu())
        #if i%cleanup_freq==0: clear_cache()
    preds = torch.stack(preds, axis=0)
    clear_cache()
    return preds

# Verified to work in every possible way! 11/20/23
class CummVar:
    """
    This CumVar class reduces across the first dimension assuming tabular data (like sklearn StandardScaler).
    If you don't want this behavior then just artificially add a new first dimension before calling on new data.
    NOTE: This is not exponentially weighted variance! That assumes distribution shift, this is just cummulative variance.
    """
    def __init__(self, correction=1):
        self.correction=correction # 1 means unbiased variance estimate
        self.SSX = 0
        self.SX = 0
        self.N = 0

    @property
    def var(self):
        var=(self.N*self.SSX-self.SX**2)/((self.N-self.correction)*self.N)
        var[var<0]=0 # ensure positivity (even with numerical instability)
        return var

    def __call__(self, X: np.ndarray):
        try:
            self.SSX += (X**2).sum(axis=0)
            self.SX += X.sum(axis=0)
            self.N += X.shape[0]
            return self.var
            # get dynamically computed running variance
        except (AttributeError, TypeError) as e:
            return self(np.atleast_1d(X))
            # If X *is just a python scalar* then convert to numpy 1d array (of length 1)

import utils

# More efficient now! It doesn't rely on pred distribution!!
# Verified to work 3/15/24
def get_BNN_pred_moments(bnn_model, x_inputs, n_samples=100, no_grad=True, verbose=True):
    if no_grad:
        bnn_model.eval()
        with torch.no_grad():
            return get_BNN_pred_moments(bnn_model, x_inputs, n_samples=n_samples, no_grad=False, verbose=verbose)

    print_interval = max(n_samples//10, 1)
    total_pred = 0
    cum_var = CummVar()
    assert n_samples>1
    for i in range(n_samples):
        if i%print_interval==0:
            if verbose:
                print(f'{i}th moment sample')
                utils.report_cuda_memory_usage(clear=False)
        pred = bnn_model(x_inputs.float())
        total_pred += pred
        cum_var(pred.unsqueeze(0)) # update, first dim is reduced so we add it
    mean_pred = total_pred/n_samples
    sd_pred = cum_var.var**0.5
    return mean_pred, sd_pred

# Verified to work: 5/20/21
def _distance(p1, p2, reduce_start_dim=2):
    if len(p2.shape)==len(p1.shape):
        print("Warning: If p1 and p2 dimensions are equal this computes *pair-wise* distances")
    squared_difference = (p1-p2)**2
    print('squared_difference.shape:', squared_difference.shape)
    squared_difference = squared_difference.reshape(*squared_difference.shape[:reduce_start_dim], -1)
    print('squared_difference.shape:', squared_difference.shape)
    distance = squared_difference.sum(axis=-1)**0.5
    print('distance.shape:', distance.shape)
    return distance


# Verified to work: 5/20/21
# TODO: Implement a technically more rigorous approach (which doesn't assume spherical predictive distribution)
# where we compute the KDE using the prediction distribution then sort based on KDE-density rather than distance from the centroid.
def find_BCI_truth_quantiles(model, inputs, outputs, n_samples=250, fake_ideal=False):
    """
    These quantiles should follow q~U(0,1) in order for BCI theory to be satisfied.
    You can simulate the ideal case as a sanity check with fake_ideal=True.
    GOTCHA: In practice with small batch and/or sample sizes even with fake_ideal=True, the distribution will not look uniform!
    GOTCHA: This is still not the most rigorous implementation! That would sort based on KDE-density rather than centroid distance.
    """
    import matplotlib.pyplot as plt
    with torch.no_grad():
        print(f'inputs.shape: {inputs.shape}')
        print(f'outputs.shape: {outputs.shape}')
        pred_distribution = get_BNN_pred_distribution(model.cuda(), inputs.cuda(), n_samples=n_samples).cuda()
        pred_center = pred_distribution.mean(axis=0)
        print('pred_distribution.shape:', pred_distribution.shape)
        print('pred_center.shape:', pred_center.shape)

        import random
        if fake_ideal: # artificially simulate the ideal case where outputs are sampled from prediction distribution
            outputs = random.choice(pred_distribution) # GOTCHA: isn't realized with small batch & sample sizes!

        # 2 for batch & pred distribution dims
        pred_distribution_distance = _distance(pred_distribution, pred_center, reduce_start_dim=2)
        error_distance = _distance(outputs.cuda(), pred_center, reduce_start_dim=1) # 1 for batch dim only

        # plot the distance distributions
        batch_display_ids = random.choices(range(inputs.shape[0]), k=5)
        for i in batch_display_ids:
            plt.hist(pred_distribution_distance[:,i].cpu(), color='blue')
            plt.axvline(error_distance[i].item(), color='red')
            plt.title(f'{i}th Distance Distribution')
            plt.show()

        distance_comparison = pred_distribution_distance<error_distance
        print('distance_comparison:', distance_comparison)
        truth_quantiles = torch.sum(distance_comparison, dim=0)/distance_comparison.shape[0]
        assert pred_distribution_distance.shape[0]==n_samples and distance_comparison.shape[0]==n_samples
        display_quantiles = list(torch.quantile(truth_quantiles, q=torch.linspace(0.0,1.0, steps=5).cuda()).cpu().numpy())
        print('1/25th quantiles of truth quantile distribution: ', display_quantiles)
        truth_quantiles=truth_quantiles.cpu()
    plt.hist(truth_quantiles)
    plt.title('Truth Quantiles (Should follow q~U(0,1))')
    plt.show()
    return truth_quantiles