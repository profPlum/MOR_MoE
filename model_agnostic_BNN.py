import os, random, warnings
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
    assert torch.isfinite(kl_loss) or not kl_loss.requires_grad, f'kl_loss={kl_loss.item()}'
    return kl_loss

_pt_nll_classificiation = torch.nn.NLLLoss(reduction='sum') # sum needed for the true NLL
nll_classification = lambda y_pred, y: _pt_nll_classificiation(y_pred, y)

# Truest NLL! (for regression)
# NOTE: This generalizes both the simpler (constant sigma) and more general (non-constant sigma) cases!
def nll_regression(y_pred, y, y_pred_sigma=1.0, reduction=torch.mean): # y_pred_sigma sigma can be assumed constant, but it's best if you don't
    y_pred_sigma = torch.as_tensor(y_pred_sigma, device=y_pred.device, dtype=y_pred.dtype)
    #element_wise_NLL = (y_pred-y)**2/(2*y_pred_sigma**2) + torch.log(y_pred_sigma) # old version
    element_wise_NLL = 0.5*((y_pred-y)/y_pred_sigma)**2 + torch.log(y_pred_sigma) # more numerically stable (don't square small sigmas b4 division)
    return reduction(element_wise_NLL) # sum (?) over element-wise NLL
    # Sum over element-wise NLL; this is mathematically correct when you assume output UQ is independent.

def nll_R2(y_pred, y, y_pred_sigma=1.0):
    our_nll = nll_regression(y_pred, y, y_pred_sigma=y_pred_sigma, reduction=torch.mean)
    baseline_nll = nll_regression(y.mean(axis=0, keepdim=True), y, y_pred_sigma=y.std(axis=0, keepdim=True), reduction=torch.mean)
    return 1-torch.exp(our_nll - baseline_nll) # = 1-exp(our_nll)/exp(baseline_nll)

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

        ## Here we make the seed for *bayesian sampling* unique across processes for better batch parallelism!
        ## NOTE: It's ugly b/c: don't want it to make the global seed unique per-process
        #pid_seed = (random.randint(0, 2**63-1) + os.getpid() + hash(os.uname().nodename)) % (2**64)
        pid_seed = (1+torch.distributed.get_rank()+torch.randint(2**63-1,size=(1,)).item()) if torch.distributed.is_initialized() else None
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

from torch.utils.data import Dataset, DataLoader
def get_dataset_size(train_dataset: Dataset | DataLoader):
    if isinstance(train_dataset, DataLoader):
        train_dataset = train_dataset.dataset
    else: assert isinstance(train_dataset, Dataset)

    X, y = train_dataset[0]
    # TODO: use model(X) instead of y to make it agnostic to classification and regression
    assert torch.is_floating_point(y), 'classification datasets are implicitly sparse which makes this fail...'
    return len(train_dataset)*y.numel()

# NOTE: this is essentially the new dnn_to_bnn() but also more versatile
# NOTE: if you pass in train_dataset and it is non-sparse (e.g. floating point regression) then it will automatically weight the get_kl_loss method!
def model_agnostic_dnn_to_bnn(dnn: nn.Module, train_dataset_size: int|Dataset|DataLoader, prior_cfg: dict = {}):
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

    original_type = type(dnn)
    # redefine class to use cached forward operation for speed & consistency
    def forward(self, *args, **kwargs):
        with parametrize.cached():
            return original_type.forward(self, *args, **kwargs)

    kl_weight=1.0
    if type(train_dataset_size) is not int:
        train_dataset_size=get_dataset_size(train_dataset_size)
    if train_dataset_size: kl_weight = 1.0/train_dataset_size
    else: warnings.warn("you didn't pass in the train_dataset_size so get_kl_loss() values will be unweighted! (make sure to weight them yourself)")
    _get_kl_loss = lambda self: get_kl_loss(self)*kl_weight
    methods =  {'forward': forward, 'get_kl_loss': _get_kl_loss} # we assign forward later to avoid edge-case
    dnn.__class__ = type(f'Bayesian{dnn.__class__.__name__}', (type(dnn),), methods)
    return dnn

# TODO: embed the moment accumulation functions into the forward method of the class
################################ BNN sampling utility functions: ################################

def clear_cache():
    ''' clear pytorch cuda cache '''
    import torch, gc
    while gc.collect(): pass
    torch.cuda.empty_cache()

try: from tqdm import tqdm
except ImportError: tqdm = lambda x: x

# Adapted to stack aleatoric moments
def get_BNN_pred_distribution(bnn_model, x_input, n_samples=100, **kwd_args):
    '''
    If you just want moments use get_BNN_pred_moments() instead as it is *much* more memory efficient (e.g. for large sample sizes).
    But this is still useful if you want an actual distribution.
    '''
    with torch.inference_mode():
        pred_distribution = [bnn_model(x_input, **kwd_args) for _ in tqdm(range(n_samples))] # list of (mu,sigma) tuples
        preds_mu = torch.stack((pred[0] for pred in pred_distribution), dim=0) # generators cost nothing
        preds_sigma = torch.stack((pred[1] for pred in pred_distribution), dim=0) # generators cost nothing
        return preds_mu, preds_sigma

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

# updated uses laws of total variance and expectation
def get_BNN_pred_moments(bnn_model, x_inputs, n_samples=100, **kwd_args):
    total_expectation = 0 # E[Y] = E[E[Y|Z]]
    total_variance = 0 # Var[Y] = E[Var[Y|Z]]+Var[E[Y|Z]]
    epistemic_variance = CummVar() # necessary for 2nd term in total variance eq.
    assert n_samples>1

    with torch.inference_mode():
        for i in tqdm(range(n_samples)):
            mu, sigma = bnn_model(x_inputs.float(), **kwd_args)
            epistemic_variance.update(mu.unsqueeze(0)) # update it, add 1st dim b/c it is reduced

            total_expectation = total_expectation + mu
            total_variance = total_variance + sigma**2

        # take average
        total_expectation = total_expectation/n_samples
        total_variance = total_variance/n_samples

        # add in the explained variance term (2nd term) = Var(mus) = Var[E[Y|Z]]
        total_variance = total_variance + epistemic_variance.var
        # NOTE: we apply Bessel's correction to 2nd term only b/c sample mean is used to estimate sample variance

        return total_expectation, total_variance**0.5

# Corrected version: uses mixture PDF before data reduction.
def log_posterior_predictive_check(model, val_data_loader, n_samples=25, use_mean=True, **kwd_args):
    model_device = next(model.parameters()).device
    assert 'cuda' in str(model_device)

    val_data_loader = tqdm(val_data_loader)
    predict = lambda X: model(X.to(model_device), **kwd_args)

    with torch.inference_mode():
        # Compute log(p(D|D_old))=log(∏_i p(D_i|D_old))=∑_i log(p(D_i|D_old))
        # NOTICE: this is log-likelihood & the sum of log-score!
        PPC_LL = torch.tensor(0.0, device=model_device)
        N_total = 0 # take average log-likelihood, more interpretable
        for X, y in val_data_loader:
            N_total += y.numel()

            # compute mixture PDF over θ samples log(p(D_i))=log(1/N*∑_j p(D_i|θ_j))
            LL_batch = -torch.inf*torch.ones_like(y, device=model_device)
            for i in range(n_samples):
                mu, sigma = predict(X)
                normal = torch.distributions.normal.Normal(mu, sigma)
                LL_batch = torch.logaddexp(LL_batch, normal.log_prob(y.to(model_device)))
            LL_batch = LL_batch-np.log(n_samples)
            PPC_LL += LL_batch.sum()
    PPC_LL = PPC_LL.item()
    if use_mean: PPC_LL /= N_total
    print(f'{PPC_LL=}', flush=True)
    return PPC_LL
