import torch
import numpy as np
from numpy.typing import NDArray
import pytorch_lightning as L
import matplotlib.pyplot as plt
import warnings
from tqdm.auto import tqdm

import scrapbook as sb
def glue_and_print(key, value):
    try: value=value.item()
    except: pass
    print(f'{key}={value}')
    sb.glue(key, value)

import JHTDB_sim_op
from JHTDB_sim_op import POU_NetSimulator, PPOU_NetSimulator

import model_agnostic_BNN # the script is now fully compatible with the current model
from model_agnostic_BNN import PredSamplingWrapper

import sys
sys.path.append('./WNO/Version_2.0.0')
sys.path.append('./IUFNO-CHL')
from glob import glob

import utils
def load_model(path, device='cuda', **kwd_args):
    ''' Wraps up all the nonsense involved in loading an inference model properly into one function. '''
    paths = glob(path)
    assert len(paths)==1, f'found these files: {paths}, but expected to find exactly one.'
    path = paths[0]
    print(f'loading model from path: {path}')

    try:
        print('loading VI model...')
        model = PPOU_NetSimulator.load_from_checkpoint(path, weights_only=False, **kwd_args)
        PredSamplingWrapper.wrap_VI_model(model)
    except Exception as e:
        try:
            print('loading deterministic model...')
            model = POU_NetSimulator.load_from_checkpoint(path, weights_only=False, **kwd_args)
        except: raise

    model = model.to(device)
    model.eval()

    print(f'num model parameters: {utils.count_parameters(model):.5e}')

    # freeze everything
    for parameter in model.parameters():
        parameter.requires_grad=False
    print('done!')
    return model

import contextlib
class SimulationFlowThroughSequence:
    ''' Data structure for indexing shifted simulation flow throughs '''
    n_steps_per_flow_thru: int = None # type: ignore

    @staticmethod
    @contextlib.contextmanager
    def flow_through_multiplier(n_flow_through_times_multiplier: int):
        cls = SimulationFlowThroughSequence # shorthand, but a static method none the less
        assert cls.n_steps_per_flow_thru % n_flow_through_times_multiplier == 0, f'{cls.n_steps_per_flow_thru=}, {n_flow_through_times_multiplier=}'
        try:
            cls.n_steps_per_flow_thru //= n_flow_through_times_multiplier
            yield
        finally: # cleanup
            cls.n_steps_per_flow_thru *= n_flow_through_times_multiplier

    # Verified to work: 7/16/26
    def continuous_index_range(self, length=None, stride=2):
        ''' Like a continuous version of range(length) for indexing shifted flow throughs (usually of this simulation).
            Also it is periodic across the flow throughs so it will always sample the same points in a flow through'''
        if length is None: length = len(self)
        return np.linspace(0, length-1, num=max(0, (length-1)*self.n_steps_per_flow_thru//stride+1), endpoint=True)

    @classmethod
    def from_output(cls, sim_data): return cls(sim_data)

    def __init__(self, sim_data):
        assert type(self.n_steps_per_flow_thru) is int, 'cls.n_steps_per_flow_thru must be specified'
        assert sim_data.shape[-1] % self.n_steps_per_flow_thru == 0, f'{sim_data.shape[-1]=}, {self.n_steps_per_flow_thru=}'
        self.full = sim_data # raw simulation data

    @property # for legacy and clarity
    def flow_thru(self): return self

    def __getitem__(self, index):
        if not -len(self) <= index <= len(self)-1: raise IndexError(f'index {index} out of bounds for flow through sequence of length {len(self)}')
        if index < 0: index += len(self) # standardize to positive indexing, e.g. -1 -> len(self)-1
        start_index = int(index*self.n_steps_per_flow_thru)
        data_slice = self.full[...,start_index:start_index+self.n_steps_per_flow_thru]
        assert data_slice.shape[-1] == self.n_steps_per_flow_thru, f'{data_slice.shape[-1]=}, {self.n_steps_per_flow_thru=}'
        return data_slice
    def __len__(self): return self.full.shape[-1]//self.n_steps_per_flow_thru

    # verified to work: 7/17/26
    def slice_flow_thru(self, start: int|None, stop: int|None=None):
        ''' start inclusive, stop exclusive like python slicing '''
        from copy import copy
        new = copy(self)
        def standardize(x):
            if x is None: return None
            if x < 0: x += len(self)
            return int(x * self.n_steps_per_flow_thru)
        new.full = self.full[...,standardize(start):standardize(stop)]
        return new

    def get_samples(self, flow_thru_index=-1, sparse=False): return self[flow_thru_index][None]

    def make_4d_sim_fig(self, vel_comp_idx:int|str='X', prefix='',
                        num_z=6, vel_component_names = ['X','Y','Z'], show=True):
        if type(vel_comp_idx) is str: vel_comp_idx = vel_component_names.index(vel_comp_idx)
        from grid_figures import GridFigure
        fig = GridFigure(f'{prefix}3d Channel Flow: {vel_component_names[vel_comp_idx]} Velocity')
        sim_data = self.full.cpu()
        viz_time_stride = 4000//self.n_steps_per_flow_thru
        for z in np.linspace(0, sim_data.shape[-2]-1, num=num_z, dtype=int):
            fig.add_3d_row(sim_data[vel_comp_idx,:,:,z], f'{z=}', x_title_func=lambda t: f't={t*viz_time_stride}',
                        img_getter=lambda array_3d, t: array_3d[:,:,t].T)
        if show: fig.show()
        return fig

class UQSimulationFlowThroughSequence(SimulationFlowThroughSequence):
    ''' Adds self.uq (with derived E_{y~N(mu, sigma)}[|y_tilde - y|] for MAP prediction)
        and self.uq.sample_moments (with original moments). '''
    def __init__(self, sim_data, uq=None, sample_moments=None):
        super().__init__(sim_data)
        self.uq = uq # nested UQ Simulation
        self.sample_moments = sample_moments # raw sample moments (for flow stats)

    @classmethod
    def from_output(cls, sim_data):
        ''' Constructs a Simulation object from the output of the model.
        If the model outputs UQ, then the nested Simulation.uq Simulation will also contain simulation uncertainty.
        And the Simulation.uq.sample_moments attribute will contain the raw sample moments '''

        sim_uq = None # default
        if type(sim_data) in (tuple, list): # handle uq
            assert len(sim_data) == 2
            sim_samples, sim_samples_uq = sim_data # unpack

            # aggregate "moments"
            sim_data = sim_samples[0]
            sim_data_uq = cls._expected_normal_MAE(sim_data, sim_samples[1:], sim_samples_uq[1:]).mean(0)
            # first calculate E_{y~N(mu, sigma)}[|y_tilde - y|] (for each epistemic mixture mode) then take expectation over mixture modes

            sample_moments = cls(sim_samples[1:], uq=cls(sim_samples_uq[1:]))
            sim_uq = cls(sim_data_uq, sample_moments=sample_moments)
        return cls(sim_data, uq=sim_uq)

    @staticmethod # verified to work: 4/16/26
    def _expected_normal_MAE(y_tilde, mu, sigma):
        ''' = E_{y~N(mu, sigma)}[|y_tilde - y|] (analytic solution) '''
        standard_normal = torch.distributions.Normal(0,1)
        alpha = (y_tilde - mu) / sigma
        phi = torch.exp(standard_normal.log_prob(alpha))
        Phi = standard_normal.cdf(alpha)
        return sigma * (2 * phi + alpha * (2 * Phi - 1))

    # verified to work: 7/17/26
    def slice_flow_thru(self, start: int|None, stop: int|None=None):
        new = super().slice_flow_thru(start, stop)
        if self.uq: new.uq = self.uq.slice_flow_thru(start, stop)
        if self.sample_moments: new.sample_moments = self.sample_moments.slice_flow_thru(start, stop)
        return new

    def get_samples(self, flow_thru_index=-1, use_MAP=False, sparse=False):
        ''' sample from sim.uq.sample_moments or return MAP prediction in compatible shape '''
        if self.uq and not use_MAP: # self.uq.sample_moments is mu, self.uq.sample_moments.uq is sigma
            pred_samples = [self.uq.sample_moments.flow_thru[flow_thru_index],
                            self.uq.sample_moments.uq.flow_thru[flow_thru_index]]
            if sparse: pred_samples = [moment[...,(0,-1)] for moment in pred_samples]
            return torch.distributions.Normal(*pred_samples).sample()
        else: return super().get_samples(flow_thru_index)

class CharacteristicTimeMSEModel:
    def __init__(self, field_tensor, n_windows=2):
        field_tensor = field_tensor.detach()
        window_size = field_tensor.shape[-1] - n_windows + 1
        print(f'window_size=field_tensor.shape[-1]-n_windows+1={window_size}')
        # ^ verified to work: 1/21/26

        # basic u0 MSE (for plotting later)
        self.u0 = field_tensor[...,0]
        self.u0_MSE = torch.vmap(torch.mean)((field_tensor.moveaxis(-1, 0)-self.u0)**2)

        MSEs = [] # MSEs[i] is the MSE of the u0 from time i to i+window_size
        from tqdm.auto import tqdm
        for i in tqdm(range(field_tensor.shape[-1]-window_size+1)):
            u0 = field_tensor[...,i]
            time_window = field_tensor.moveaxis(-1, 0)[i:i+window_size]
            MSEs.append(torch.vmap(torch.mean)((time_window-u0)**2))
        MSEs = torch.stack(MSEs, dim=0)

        import scipy.optimize as optimize
        t = np.arange(window_size)
        def exp_error_residual(x):
            C, t_c = x # this is what the array means
            pred_MSE = C*(1-np.exp(-t/t_c))
            return (abs(MSEs-pred_MSE)).mean().item()
        opt_result = optimize.shgo(exp_error_residual, bounds=((1e-16, 100), (1, 500)))
        #opt_result = optimize.minimize(exp_error_residual, x0=np.random.uniform(1, 250, size=2))
        self.asymptotic_MSE, self.characteristic_time = opt_result.x
        print(f'{opt_result=}')
        print(f'characteristic_time=t_c={self.characteristic_time}, asymptotic_MSE=C={self.asymptotic_MSE}')

    def predict_MSE(self, t):
        return self.asymptotic_MSE*(1-np.exp(-t/self.characteristic_time))

    def plot_MSE_vs_pred(self, other_model=None, other_model_label='other'):
        print(f'characteristic_time=t_c={self.characteristic_time}, asymptotic_MSE=C={self.asymptotic_MSE}')
        t = np.arange(self.u0_MSE.shape[0])
        if other_model:
            plt.plot(other_model.u0_MSE, label=other_model_label+'.u0_MSE')
            plt.plot(other_model.predict_MSE(t), label=other_model_label+'.prediction')
        plt.plot(self.u0_MSE, label='$MSE(u_0,u_t)$')
        plt.plot(self.predict_MSE(t), label='prediction')
        plt.axvline(self.characteristic_time, color='k', linestyle='--', label=f't_c={self.characteristic_time}')
        plt.axhline(self.asymptotic_MSE, color='k', linestyle=':', label=f'C={self.asymptotic_MSE}')
        plt.legend()
        plt.title('MSE of u0 vs time')
        plt.show()

#########################################################
# Cross-Correlation Diagnostics
#########################################################

import pandas as pd
def self_xcor(flow_data) -> NDArray:
    ''' Takes the X-cross-correlation of the flow data with itself beginning vs ending of the flow.
    Assumes flow_data.shape==(3,Nx,Ny,Nz,times) '''
    f1 = flow_data[...,0]
    f2 = flow_data[...,-1]

    # * .mean(0,2,3) can be on the outside of the irfft but it is moved inside to make the irfft more efficient
    # * and iFFT(FFT(X).conj()*FFT(Y)) = cross_correlation(X,Y) (essentially template matching by sliding one across the other
    # * .conj() ensures similar signal maximizes the real component (x-iy)(x+iy)=x^2-(iy)^2=x^2+y^2 (also is necessary to avoid the "flip" that happens in convolution)
    # * field - field.mean(1, keepdims=True) centers each signal like how you center variables when taking correlation (same with scale normalization)
    def x_normalize_spatial_field(field):
        ''' Compute zero-mean signals along x, then get norms for normalization. '''
        field = field - field.mean(1, keepdims=True) # mu_x=0
        field_std = np.sqrt((field**2).mean(1, keepdims=True))
        field /= np.where(field_std == 0, 1, field_std) # sigma_x=0 & avoid division by zero
        return field # normalized field
    f1 = x_normalize_spatial_field(f1)
    f2 = x_normalize_spatial_field(f2)
    cc_fft = np.fft.rfft(f1, axis=1).conj() * np.fft.rfft(f2, axis=1) # cross-correlation in Fourier space
    return np.fft.irfft(cc_fft.mean((0,2,3)))/f1.shape[1] # average over y & z, then normalize by Nx to get analog of Pearson's R (range in [-1,1])

MSE = lambda pred, true: float(np.mean((np.asarray(pred) - np.asarray(true))**2))

def xcor_metrics(pred_xcor, true_xcor):
    return pd.Series({'xcor_mse': MSE(pred_xcor, true_xcor), 'xcor_peak_magnitude_delta': pred_xcor.max()-true_xcor.max(),
                      'xcor_peak_loc_delta': pred_xcor.argmax()-true_xcor.argmax()}, dtype='float32')

def glue_and_print_metrics(metrics: pd.Series, postfix=''):
    for name in metrics.index:
        glue_and_print(f'{name}{postfix}', metrics[name])

# NOTE: peaks indicate "template matches", strength of match is given by peak amplitude
# The rest of the signal is still useful for MSE but not necessary to aggregate
def cross_correlation_comparison(pred_flow, real_channel_flow, title='', plot=True):
    ''' This code assumes you have 2 numpy arrays loaded in memory:
    pred_samples with shape, (3,Nx,Ny,Nz,times)
    real_channel_flow with shape, (3,Nx,Ny,Nz,times)
    returns: the MSE between the xcor of the pred and true'''
    assert tuple(pred_flow.shape)==tuple(real_channel_flow.shape)

    pred_xcor = self_xcor(pred_flow)
    true_xcor = self_xcor(real_channel_flow)
    metrics = xcor_metrics(pred_xcor, true_xcor)
    if plot:
        plt.plot(true_xcor,label='true')
        plt.plot(pred_xcor,label='model')
        plt.axvline(np.argmax(pred_xcor), color='orange', linestyle='--', label='model peak')
        plt.axvline(np.argmax(true_xcor), color='blue', linestyle='--', label='true peak')
        plt.legend()
        plt.title('Flow Cross-Correlation: '+title)
        plt.show()
        print('MSE between xcor of pred and true:', metrics['xcor_mse'])
    return metrics

def cross_correlation_comparison_cumulative(sim, real_channel_flow, flow_thru_index=-1,
                                            n_xcor_steps=25, beta=0.15, should_plot=True):
    ''' n_xcor_steps = 25 should work with the time strides we've tested: 4, 8, and 16
        beta is the weight for the previous metrics vs the new metrics for EMA '''
    plot_interval = n_xcor_steps//5

    assert sim.flow_thru[flow_thru_index].shape[-1] == real_channel_flow.shape[-1]
    metrics_cum: pd.Series = None # bias-corrected EMA requires direct assignment for the first iteration
    end_steps = np.linspace(0, real_channel_flow.shape[-1], num=n_xcor_steps+1, dtype=int)[1:]
    for i, end_step in enumerate(end_steps):
        i+=1 # 1-based indexing
        should_plot_i = should_plot and (i % plot_interval == 0 or i==n_xcor_steps)
        metrics_i = cross_correlation_comparison(sim.flow_thru[flow_thru_index][...,:end_step],
                    real_channel_flow[...,:end_step], plot=should_plot_i,
                    title=f'$T_0$ vs $T_0+{i/n_xcor_steps}$')
        if metrics_cum is None: metrics_cum = metrics_i.copy()
        else: metrics_cum = beta*metrics_cum + (1-beta)*metrics_i
        if should_plot_i:
            print(f'{end_step=}')
            print('='*75)
    return pd.concat([metrics_cum.add_suffix('_cum'), metrics_i.add_suffix('_last')])

#########################################################
# Flow Statistics & Diagnostics
#########################################################

'''
This code assumes you have 2 numpy arrays loaded in memory:
    pred_samples with shape, (samples,3,Nx,Ny,Nz,times) (from 1 to 2 flow through times, it can be any flow through time though)
    real_channel_flow with shape, (3,Nx,Ny,Nz,times)
'''

def E1d(u, epsilon_multiplier=1.0, nk=30, Lx=8*np.pi, Lz=4*np.pi): # & Ly=2
    '''
    arguments:
        u: input function, u.shape==(channels=3,Nx,Ny,Nz)
        epsilon_multiplier: for width of ring to project to point
        nk: number of points along radius to project on to
        Lx,Lz: domain lengths (default values are for the original dataset size)
    output: energy spectrum from u. E1d.shape==[k, y]:
        First axis is the energy spectrum. Second is y coordinate
    '''
    if len(u.shape)!=4 or u.shape[0]!=3:
        raise ValueError(f'expected (channels=3,Nx,Ny,Nz), got {u.shape}')
    assert tuple(u.shape)==(3, 70, 17, 52) # TODO: delete me
    u = np.moveaxis(np.asarray(u), 0, -1) # original code requires: u.shape==(Nx,Ny,Nz,3)

    def E(u): # energy in Fourier space (2D)
        uh = np.fft.rfftn(u,axes=[0,2])
        return np.sum(np.abs(uh)**2,axis=-1)
    def npmap(f,a): # map function over array
        return np.asarray(list(map(f,a)))

    nx,ny,nz = u.shape[0:-1]
    kx = np.fft.fftfreq(nx,d=Lx/nx)  * 2 * np.pi # 2 * np.pi "converts to angular wavenumbers"?
    kz = np.fft.rfftfreq(nz,d=Lz/nz)  * 2 * np.pi # ^ But not sure if we need it or not...
    dk = np.sqrt(kx[1]**2 + kz[1]**2)
    epsilon = epsilon_multiplier*dk
    Kxz = np.stack(np.meshgrid(kx,kz,indexing='ij'),axis=-1)
    K = np.sqrt(Kxz[...,0]**2 + Kxz[...,1]**2)
    k = np.linspace(0,min(np.max(kx),np.max(kz)),nk)
    Eu = E(u)

    return k,2.*np.transpose(npmap(
        lambda j:npmap(
            lambda ki:np.sum(Eu[:,j][np.abs(K-ki)<epsilon]),
            k),
        np.arange(ny)))

def get_last_TS_energy_spectra(flow_samples, **kwd_args):
    '''
    meant to be an intuitive wrapper for E1d that works for both single and multiple flow samples
    flow_samples: (samples?,3,Nx,Ny,Nz,times)
    **kwd_args: additional arguments for E1d
    returns: (k,Es) where Es.shape==(samples?,nk)
    '''
    try:
        Es = []
        for samp in flow_samples:
            # samp.shape==(3,Nx,Ny,Nz,time)
            last_TS = samp[..., -1]
            k,EE1 = E1d(last_TS, **kwd_args)
            Es.append(EE1)
        Es=np.asarray(Es if len(Es)>1 else Es[0])
        return k,Es
    except ValueError as e:
        assert 'expected (channels=3,Nx,Ny,Nz)' in str(e), 'Got unexpected ValueError: '+str(e)
        return get_last_TS_energy_spectra([flow_samples], **kwd_args)

#########################################################
# AI-generated epsilon multiplier calibration code
#########################################################

def _plot1d_epsilon_scale(spatial_shape):
    ''' Same resolution scaling as plot_1dDiagnostics (1.0 was for the original grid). '''
    return float(np.mean(np.array([103, 26, 77]) / np.asarray(spatial_shape)))

# TODO: try more windows (e.g. 25)
def energy_spectrum_windowed_cv(flow, epsilon_multiplier=1.0, n_windows=10, k_trim=5, **e1d_kwargs):
    ''' Split one flow-through into n_windows over time; CV = std/mean of E(κ) across windows.
        CV is averaged over κ-bins with index >= k_trim (Ravi's "κ>=5" = skip first 5 bins).
        Returns (k, Es, cv, mean_cv) with Es.shape==(n_windows, nk) at midplane y. '''
    flow = np.asarray(flow)
    assert flow.ndim == 5 and flow.shape[0] == 3, f'expected (3,Nx,Ny,Nz,T), got {flow.shape}'

    # TODO: delete maybe? I'm just keeping it for now to see how far off the original default was
    eps = epsilon_multiplier * _plot1d_epsilon_scale(flow.shape[1:4])
    y_index = flow.shape[2] // 2
    seq = SimulationFlowThroughSequence(flow) #.slice_flow_thru(-1) # get the last flow-through only
    assert len(seq) == 1
    with SimulationFlowThroughSequence.flow_through_multiplier(n_windows):
        assert len(seq) == n_windows, f'expected {n_windows} windows, got {len(seq)} (check n_steps divisible by n_windows)'
        k, Es = get_last_TS_energy_spectra(seq, epsilon_multiplier=eps, **e1d_kwargs)
        Es = Es[..., y_index]
    mu = Es.mean(0)
    cv = Es.std(0, ddof=1) / np.maximum(mu, 1e-30) # CV = std/mean, & np.maximum avoids division by zero
    assert k_trim < len(cv), 'k_trim must be less than the number of κ-bins'
    return cv[k>=k_trim]

def calibrate_epsilon_multiplier(flow, epsilon_multipliers=np.linspace(0.25, 16, 50), n_windows=10, k_trim=5,
                                 cv_target=0.05, use_max_cv=True, should_plot=True, **e1d_kwargs):
    ''' Calibrate epsilon_multiplier so [mean|max](std/mean of E) over temporal windows ≤ cv_target
        for κ-bins with index >= k_trim (Ravi: try for <5% on κ>=5).
        flow: one prediction (or real) flow-through, shape (3,Nx,Ny,Nz,T).
        Returns (recommended_epsilon_multiplier, summary DataFrame). '''
    rows = []
    for eps in tqdm(np.asarray(epsilon_multipliers, dtype=float), desc='calibrate ε'):
        cv = energy_spectrum_windowed_cv(flow, eps, n_windows, k_trim=k_trim, **e1d_kwargs)
        rows.append({'epsilon_multiplier': eps, 'mean_cv': cv.mean(), 'max_cv': cv.max()})
    df = pd.DataFrame(rows)
    thresholded_cv = 'max_cv' if use_max_cv else 'mean_cv'
    ok = df[df[thresholded_cv] <= cv_target]
    # smallest ε that meets the target (least shell smoothing among stable choices)
    if len(ok): recommended = float(ok['epsilon_multiplier'].iloc[0])
    else:
        # NOTE: should be the same as just giving the highest epsilon_multiplier tested
        recommended = float(df.loc[df[thresholded_cv].idxmin(), 'epsilon_multiplier'])
        print(f'warning: no ε reached {thresholded_cv}≤{cv_target}; using lowest-CV ε={recommended}')
    print(df.head().to_string(index=False))
    print(f'recommended epsilon_multiplier={recommended} (target {thresholded_cv}≤{cv_target} for κ≥{k_trim})')
    if should_plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.plot(df['epsilon_multiplier'], df[thresholded_cv], 'o-')
        ax.axhline(cv_target, color='C1', ls='--', label=f'target={cv_target}')
        ax.axvline(recommended, color='k', ls=':', label=f'rec={recommended:g}')
        #ax.set_xscale('log')
        ax.set_xlabel('epsilon_multiplier')
        ax.set_ylabel(rf'{thresholded_cv} of $E(\kappa)$ $\forall\kappa\geq{k_trim}$')
        ax.legend(fontsize='x-small')
        fig.tight_layout()
        plt.show()
        plt.close()
    return recommended, df

#########################################################

#real_channel_flow.shape==(3,Nx,Ny,Nz,times)
#pred_samples.shape==(samples,3,Nx,Ny,Nz,times)
def plot_1dDiagnostics(pred_samples, real_channel_flow, k_trim=2, epsilon_multiplier=1.0,
                       should_plot=True, reduce_y_line=False, **kwd_args): # takes ~200ms
    ''' reduce_y_line: if True, then the y-line is reduced by averaging over x and z dimensions else the y-line is chosen arbitrarily
        k_trim: trim the first k_trim points from the energy spectrum (to prevent them from dominating)
        epsilon_multiplier: multiplier for the epsilon radius of the energy spectrum (to prevent aliasing)
        **kwd_args: additional arguments for E1d() '''
    assert pred_samples.shape[1:-1]==real_channel_flow.shape[:-1] and \
        pred_samples.shape[-1] in {real_channel_flow.shape[-1], 2}, \
        f'invalid shapes: {pred_samples.shape[1:]=}, {real_channel_flow.shape=}'

    CI_coef = 1.96 # 95% CI
    metrics = {} # all 1d metrics

    stride = np.array([103,26,77])/real_channel_flow.shape[1:4] # approximate the stride of the flow
    epsilon_multiplier *= np.mean(stride) # 1 was the default for the original dataset size

    with warnings.catch_warnings(): # Safely ignore DoF warning (for np.std with only MAP/MLE sample)
        warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

        k,EE = get_last_TS_energy_spectra(real_channel_flow, epsilon_multiplier=epsilon_multiplier, **kwd_args)
        k,Es = get_last_TS_energy_spectra(pred_samples, epsilon_multiplier=epsilon_multiplier, **kwd_args)

        # we are trimming the k_trim because they contain so much energy that they distort the plots
        y_index = real_channel_flow.shape[2]//2 # Dwyer: should be the midpoint b/c it avoids the walls
        Es_y = Es[:,k_trim:,...,y_index] # NOTE: tested on 7/20/26 that len(Es.shape)==3 which would make the "..." redundant...
        mu = Es_y.mean(0)
        std = Es_y.std(0)
        k,EE = k[k_trim:],EE[k_trim:,y_index]
        metrics['mse_log_energy_spectrum'] = MSE(np.log(Es_y), np.log(EE))
        if should_plot:
            y = np.loadtxt('y.txt') # Dwyer: we need to interpolate this to the number of y points in the flow
            y = np.interp(np.linspace(0,1,real_channel_flow.shape[2]), np.linspace(0,1,len(y)), y) # interp(x, xp, fp)
            fig,ax = plt.subplots(1,4,figsize=(8,2),sharex='col',sharey='col')
            ax[0].plot(k,EE,'C1') # NOTE: C1 & C2 are colors
            ax[0].plot(k,mu,'--k')
            ax[0].fill_between(k,mu-CI_coef*std,mu+CI_coef*std)
            ax[0].plot(k,3e4*k**(-5./3.),'C2')
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            #ax[0].set_ylim(2e3,5e4)
            ax[0].set_ylabel(r'$E(\kappa)$')
            ax[0].set_xlabel('$\kappa$')
            #ax[0].title.set_text('Energy Spectrum')

        # TODO: should probably not reduce channels at all for the MSE metric (just for plotting)
        # TODO: should probably take the L2 norm over channels instead of mean (to measure vel-vec length)
        # TODO: replace xz_index with : then take mean over x and z

        # get_y_line: get the y-line of the array (by reducing x and z dimensions)
        if reduce_y_line: get_y_line = lambda arr: arr[...,-1].mean(axis=(-3,-1))
        else: get_y_line = lambda arr, xz_index = 10: arr[...,xz_index,:,xz_index,-1] # Dwyer: why 10? <-- apparently arbitrary?
        pred_samples_y_line = get_y_line(pred_samples)
        real_channel_flow_y_line = get_y_line(real_channel_flow)
        res = pred_samples_y_line.mean(1) # mean over channels
        mu = res.mean(0)
        std = res.std(0)
        true_u1 = real_channel_flow_y_line.mean(0) # mean over channels
        metrics['mse_bulk_velocity'] = MSE(res, true_u1)
        if should_plot:
            ax[1].plot(y,true_u1,'C1')
            ax[1].plot(y,mu,'--k')
            ax[1].fill_between(y,mu-CI_coef*std,mu+CI_coef*std)
            ax[1].set_ylabel('$u_{1}$')
            ax[1].set_xlabel('$y$')
            #ax[1].title.set_text('Bulk Velocity')

        res = np.sqrt(pred_samples_y_line.var(1))
        mu = res.mean(0)
        std = res.std(0)
        true_urms = np.sqrt(real_channel_flow_y_line.var(0))
        metrics['mse_rms'] = MSE(res, true_urms)
        if should_plot:
            ax[2].plot(y,true_urms,'C1')
            ax[2].plot(y,mu,'--k')
            ax[2].fill_between(y,mu-CI_coef*std,mu+CI_coef*std)
            ax[2].set_ylabel('$u_{rms}$')
            ax[2].set_xlabel('$y$')
            #ax[2].title.set_text('RMS')

        true_xcor = self_xcor(real_channel_flow)
        pred_xcor = [self_xcor(sample) for sample in pred_samples]
        p_xcor_mu = np.mean(pred_xcor,0)
        p_xcor_std = np.std(pred_xcor,0)
        xcor_metrics_ = sum(xcor_metrics(p_xcor_i, true_xcor) for p_xcor_i in pred_xcor)/len(pred_xcor)
        metrics = pd.concat([pd.Series(metrics), xcor_metrics_])
        if should_plot:
            ax[3].plot(pred_xcor[0],'--k', label='model')
            for i in range(1,len(pred_xcor)):
                ax[3].plot(pred_xcor[i],'--k')
            ax[3].plot(p_xcor_mu,'r',label='mu')
            ax[3].plot(true_xcor,'C1',label='true')
            #ax[3].fill_between(np.arange(len(p_xcor_mu)), p_xcor_mu-CI_coef*p_xcor_std, p_xcor_mu+CI_coef*p_xcor_std)
            #ax[3].title.set_text('Xcor')
            plt.legend(fontsize='xx-small', loc='best')
            ax[3].set_xlabel('$x/\Delta x$')
            ax[3].set_ylabel('$u(0) \star u(T)$')
            fig.tight_layout()

            print('='*75)
            plt.show()
            plt.close()
            print(metrics)
    return metrics