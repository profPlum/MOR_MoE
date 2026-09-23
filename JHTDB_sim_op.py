import warnings

import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L

from lightning_utils import *
from POU_net import POU_net, PPOU_net
from JHTDB_data_loading import JHTDBDataModule

_rfft = functools.partial(torch.fft.rfftn,dim=[0,1,2])
_irfft = functools.partial(torch.fft.irfftn,dim=[0,1,2])

def _divide_no_nan(a,b):
    #return a/b w/o nan values or gradient
    mask = b!=0
    b = b + ~mask #aka b[~mask] = 1
    result = a/b
    mask = torch.broadcast_to(mask, result.shape)
    clean_result = torch.zeros_like(result)
    clean_result[mask] = result[mask]
    return clean_result

# original, fails with gradients for complex types
#def divide_no_nan(a,b):
#    return torch.nan_to_num(a/b,nan=0.0, posinf=0., neginf=0.)

class IdentityOp:
    def forward(self, X):
        return X

# Private to force access through (P)POU_NetSimulator.Sim
class _Sim(L.LightningModule):
    '''
    Raw Sim[ulator] class that solves naiver stokes with learned model correction.
    We wrapped Dr. Patel's original code to do axis swapping
    (the code needs channel dim last but pytorch needs it right after batch dim),
    in a way that is *compatible with vmap* for batching!!
    '''
    n_state_channels = 3 # u, v, w
    def __init__(self,nx=103,ny=26,nz=77,Lx=8*np.pi,Ly=2.0,Lz=3*np.pi,nu=5e-5,dt=0.0065,
                 u_b: torch.Tensor=0, use_PDE_solver=True, anisotropic_filter: bool=False, disable_filter: bool=False,
                 dealias_before_quadratic: bool=False, apply_pde_filter_bottleneck: bool=True, n_inputs: int=1):
        ''' Defaults are set to the values needed for JHTDB channel flow.
            Also note that nu:=viscosity, Lx,Ly,Lz:=domain dimensions (physical),
            and nx,ny,nz:=grid dimensions (virtual)
            u_b:= real_channel_flow[0].mean((0,2,3)) (mean x velocity across the y-dimension)
            passing a non-zero u_b will automatically enable manual advection (aka forcing)
            anisotropic_filter:= if True, apply per-dimension 2/3 dealiasing in index space (recommended for anisotropic Lx,Ly,Lz);
                                if False, use the legacy isotropic |k|-ball cutoff. Ignored if disable_filter=True.
            disable_filter:= if True, do not apply any spectral truncation (filt and filt2 are all-ones).
            dealias_before_quadratic:= if True, apply filt before forming the quadratic term u*u (reduces aliasing).
            apply_pde_filter_bottleneck:= if True, apply filt to every PDE step (legacy low-pass bottleneck).'''
        assert n_inputs >= 1, 'n_inputs must be at least 1'
        super().__init__()
        self.n_inputs = n_inputs
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nu = nu
        self.use_PDE_solver = use_PDE_solver # whether to use the PDE solver
        self.dealias_before_quadratic = dealias_before_quadratic
        self.apply_pde_filter_bottleneck = apply_pde_filter_bottleneck

        self.u_b = torch.as_tensor(u_b) # u_b = real_channel_flow[0].mean((0,2,3)) (for manual advection term)
        self.use_manual_advection = torch.any(self.u_b != 0) # whether to use the manual advection term
        assert self.u_b.numel() in [1, ny]

        self.k = torch.as_tensor(np.stack(np.meshgrid(np.fft.fftfreq(nx)*nx*2.*np.pi/Lx,
                                       np.fft.fftfreq(ny)*ny*2.*np.pi/Ly,
                                       np.fft.rfftfreq(nz)*nz*2.*np.pi/Lz,indexing='ij'),axis=-1)).cfloat()

        ## not used
        #self.x = torch.as_tensor(np.stack(np.meshgrid(np.arange(nx)/nx*Lx,
        #                                   np.arange(ny)/ny*Ly,
        #                                   np.arange(nz)/nz*Lz,indexing='ij'),axis=-1))

        ## not used
        #self.xi = (self.x[...,0]>=np.pi/4)*(self.x[...,0]<=Lx-np.pi/4)

        self.knorm2 = torch.sum(self.k**2,-1).real.float()
        self.Ainv =  torch.as_tensor(1./(1.+nu*np.einsum('...j,...j->...',self.k,self.k)))
        self.shapef = [nx,ny,nz]
        self.shapeh = [nx,ny,nz//2+1]

        if disable_filter:
            filt = torch.ones(self.shapeh, dtype=torch.float32)
            self.filt2 = torch.ones(self.shapeh, dtype=torch.float32)
        elif anisotropic_filter:
            print('Using anisotropic filter')
            # Dealiasing/truncation in *index space* (2/3-rule per dimension).
            # This avoids unit-mismatch issues when Lx,Ly,Lz are anisotropic (e.g. Ly=2 makes ky spacing large in physical units).
            kx_idx = np.fft.fftfreq(nx) * nx
            ky_idx = np.fft.fftfreq(ny) * ny
            kz_idx = np.fft.rfftfreq(nz) * nz
            KX, KY, KZ = np.meshgrid(kx_idx, ky_idx, kz_idx, indexing='ij')

            filt = torch.as_tensor(
                (np.abs(KX) <= nx/3) & (np.abs(KY) <= ny/3) & (np.abs(KZ) <= nz/3)
            )
            self.filt2 = torch.as_tensor(
                (np.abs(KX) <= nx/6) & (np.abs(KY) <= ny/6) & (np.abs(KZ) <= nz/6)
            )
        else:
            print('Using isotropic (legacy) filter')
            # Legacy isotropic cutoff in physical |k|. Note: threshold is in grid-count units, so this can be overly
            # restrictive when Lx,Ly,Lz are anisotropic.
            filt = torch.as_tensor((torch.sqrt(self.knorm2) <= 2./3*(min(self.nx,self.ny,self.nz)/2+1))) # only used locally
            self.filt2 = torch.as_tensor((torch.sqrt(self.knorm2) <= 1./3*(min(self.nx,self.ny,self.nz)/2+1)))

        self.filt = filt
        if self.apply_pde_filter_bottleneck:
            self.Ainv = self.Ainv * self.filt.to(self.Ainv.dtype)
        self.dt = dt
        self.dx = Lx/nx
        #self.dy = Ly/ny
        #self.dz = Lz/nz
        #self.forcing = 0.*self.k # not used
        #self.forcing[4,4,4,0] = 10. # not used

        # sanity checks: these filters act in Fourier space (rfft over last dim)
        assert tuple(self.filt.shape) == tuple(self.shapeh), f"Unexpected filt shape {tuple(self.filt.shape)} vs {tuple(self.shapeh)}"
        assert tuple(self.filt2.shape) == tuple(self.shapeh), f"Unexpected filt2 shape {tuple(self.filt2.shape)} vs {tuple(self.shapeh)}"

        #self.eta = 1e-3 # not used
        #self.nu_num = 1e-3 # not used
        self.op = IdentityOp() # identity by default
        self.vmap_NSupd = torch.vmap(self.NSupd) # only this needs vmapping, NeuralOp is already batched

        for name, value in vars(self).copy().items():
            if isinstance(value, torch.Tensor):
                del vars(self)[name]
                self.register_buffer(name, value.detach(), persistent=False)

    @classmethod # construct a Sim object from a JHTDBDataModule
    def from_JHTDB_data_module(cls, data_module: JHTDBDataModule, use_manual_advection=False, **kwd_args):
        """Build Sim from data module. All kwd_args are forwarded to the constructor."""
        merged = data_module.meta_data().simulator_kwd_args | kwd_args
        if use_manual_advection: merged['u_b'] = data_module.u_b
        return cls(**merged)

    def genIC(self, from_LES=False):
        C = self.n_state_channels
        h = torch.tensor(np.random.normal(0,1,(self.nx,self.ny,self.nz,C))).float().to(self.device)
        hh = _rfft(h) * self.filt2[...,None]
        proj = self.k*(torch.sum(self.k*hh,axis=-1)/self.knorm2)[...,None]
        proj[0]=0
        u0 = _irfft(hh - proj, s=self.shapef)
        u0 = u0.permute(-1,0,1,2) # [C, X, Y, Z] (i.e. torch.moveaxis(u0,-1,0))
        if self.n_inputs > 1:
            u0 = u0.repeat(self.n_inputs, 1, 1, 1) # [u,v,w] copied across the input window
        if from_LES:
            assert self.op is not IdentityOp, 'Cannot use LES IC with IdentityOp'
            with torch.inference_mode():
                # 20 keeps the previous realism burn-in; n_inputs replaces every copied frame
                traj = self.evolve(u0, n=max(20, self.n_inputs), intermediate_outputs=True)
            if isinstance(traj, tuple): traj = traj[0] # drop UQ trajectory, TODO: find better way to handle this
            frames = traj[..., -self.n_inputs:].moveaxis(-1, 0) # [n_inputs, C, X, Y, Z]
            u0 = frames.reshape(-1, *frames.shape[2:])
        return u0

    def _as_history(self, u):
        ''' u is [B, C, X, Y, Z]. A single frame is repeated to fill the input window. '''
        C = self.n_state_channels
        if u.shape[1] == C and self.n_inputs > 1:
            return u.repeat(1, self.n_inputs, 1, 1, 1)
        assert u.shape[1] == self.n_inputs * C, f'expected {self.n_inputs * C} channels, got {u.shape[1]}'
        return u

    def _model_input(self, history):
        ''' PDE-update only the newest frame, then stack it behind the older frames. '''
        C = self.n_state_channels
        latest = self.vmap_NSupd(history[:, -C:])
        if self.n_inputs == 1: return latest
        return torch.cat([history[:, :-C], latest], dim=1) # keep older frames, replace the newest

    def _advance_history(self, history, new_state):
        if self.n_inputs == 1: return new_state
        return torch.cat([history[:, self.n_state_channels:], new_state], dim=1)

    # NOTE: u.shape==[channel, x, y, z]
    def _NSupd(self,u): # Navier-stokes update
        u = u.permute(1,2,-1,0) #torch.moveaxis(u, 0, -1)
        uh = _rfft(u)
        assert list(uh.shape)[:-1]==self.shapeh
        u_for_quadratic = u
        if self.dealias_before_quadratic:
            u_for_quadratic = _irfft(uh * self.filt[...,None].to(uh.dtype), s=self.shapef)
        u2h = _rfft(torch.einsum('...i,...j->...ij',u_for_quadratic,u_for_quadratic))
        u = _irfft(self.Ainv[...,None]*(
            uh + self.dt*(-1.j*torch.einsum('...j,...ij->...i',self.k,u2h)
                 + 1.j*_divide_no_nan(torch.einsum('...i,...j,...k,...jk->...i',self.k,self.k,self.k,u2h),self.knorm2[...,None])
                 )),
                 s=self.shapef
            )
        return u.permute(-1,0,1,2) # i.e. torch.moveaxis(u, -1, 0)

    def NSupd(self,u):
        if not self.use_PDE_solver: return u
        u_new = self._NSupd(u)
        if self.use_manual_advection:
            u_new = u_new + self.manual_advection_term(u)
        return u_new

    # set the neural operator for correction
    def set_operator(self, op):
        self.op = op

    # NOTE: u.shape==[channel, x, y, z]
    # verified to work: 1/23/26
    def manual_advection_term(self, u):
        dudx = (u.roll(-1, dims=1)-u.roll(1, dims=1))/(2*self.dx)

        # u_b = real_channel_flow[0].mean((0,2,3))
        assert self.u_b.ravel().shape[0]==u.shape[2]
        self.u_b = self.u_b.reshape(1, 1, -1, 1) # make y-col vector
        return -self.dt * self.u_b * dudx

    # This needs to output intermediate time-steps to get full loss!
    def evolve(self,u0,n,intermediate_outputs=False, intermediate_output_stride=1, to_cpu=False):
        added_batch = len(u0.shape)==4 # [C,X,Y,Z] -> [1,C,X,Y,Z]; vmap expects a batch dim
        history = self._as_history(u0[None] if added_batch else u0)
        outputs = []
        new_state = history[:, -self.n_state_channels:]
        for i in range(n):
            new_state = self.op.forward(self._model_input(history))
            history = self._advance_history(history, new_state)
            if i%100==0 and new_state.isnan().any():
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            if intermediate_outputs and i%intermediate_output_stride==0:
                outputs.append(new_state.to('cpu', non_blocking=True) if to_cpu else new_state)

        if intermediate_outputs and to_cpu:
            torch.cuda.synchronize() # async sync

        # (tl;dr: stack(axis=-1)) to_cpu is used for evaluation which has the pattern of indexing individual timesteps and this helps OS paging
        fancy_stack = lambda outputs: torch.stack(outputs,axis=0).moveaxis(0,-1) if to_cpu else torch.stack(outputs,axis=-1)

        # time dim is the last dim (if it exists). Record the 3-channel state, not the stacked window.
        outputs = fancy_stack(outputs) if intermediate_outputs else new_state
        return outputs.squeeze() if added_batch else outputs

# For use with PPOU_net. Uncertainty is predicted each step and is not fed back in.
class _UQ_Sim(_Sim):
    def evolve(self,u0,n,intermediate_outputs=False, intermediate_output_stride=1, to_cpu=False):
        added_batch = len(u0.shape)==4
        history = self._as_history(u0[None] if added_batch else u0)
        u_outputs, uq_outputs = [], []
        new_state = history[:, -self.n_state_channels:]
        new_uq = torch.zeros_like(new_state)
        for i in range(n):
            # omit the uncertainty input; PPOU_net.forward fills it with zeros matching the velocity window
            new_state, new_uq = self.op.forward(self._model_input(history))
            history = self._advance_history(history, new_state)
            if i%100==0 and (new_state.isnan().any() or new_uq.isnan().any()):
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            if intermediate_outputs and i%intermediate_output_stride==0:
                u_outputs.append(new_state.to('cpu', non_blocking=True) if to_cpu else new_state)
                uq_outputs.append(new_uq.to('cpu', non_blocking=True) if to_cpu else new_uq)

        maybe_squeeze = lambda output: output.squeeze() if added_batch else output
        fancy_stack = lambda outputs: torch.stack(outputs,axis=0).moveaxis(0,-1) if to_cpu else torch.stack(outputs,axis=-1)

        if intermediate_outputs and to_cpu:
            torch.cuda.synchronize() # async sync

        if intermediate_outputs:
            return maybe_squeeze(fancy_stack(u_outputs)), maybe_squeeze(fancy_stack(uq_outputs))
        return maybe_squeeze(new_state), maybe_squeeze(new_uq)

class POU_NetSimulator(POU_net):
    ''' Combines the POU_net with the raw Sim[ulator] class (internally). '''
    Sim=_Sim # Sim class for this class (e.g. Sim or Sim_UQ)
    def __init__(self, *args, n_steps: int, simulator_kwd_args: {}, **kwd_args):
        super().__init__(*args, **kwd_args)
        self.simulator = self.Sim(**simulator_kwd_args)
        assert issubclass(self.Sim, _Sim) # should be a descendant of Sim (sanity check)
        self.simulator.set_operator(super()) # this will internally call super().forward(X)
        self.n_steps = n_steps # n timesteps for PDE evolution

    def forward(self, X, n_steps: int=None, intermediate_outputs=True, **kwd_args):
        #NOTE: X.shape==[batch, channel, x, y, z]

        # by caching the gating weights we optimize memory & time
        # also it is safe because there are no optimization steps inside a forward!
        with self.gating_net.cached_gating_weights():
            if n_steps is None: n_steps=self.n_steps
            return self.simulator.evolve(X, n=n_steps, intermediate_outputs=intermediate_outputs, **kwd_args)
            # evolve has now been vmapped internally!

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch # y.shape==[batch, channel, x, y, z, time]

        try:
            org_steps=self.n_steps
            self.n_steps = y.shape[-1]
            super().validation_step(batch, batch_idx, dataloader_idx)
        finally:
            self.n_steps=org_steps

    def training_step(self, batch, batch_idx=None, val=False):
        loss=super().training_step(batch, batch_idx=batch_idx, val=val)
        assert self.training == (not val)
        if not loss.isfinite() and self.training: raise RuntimeError('NaN loss! aborting training')
        # be careful! training_step is used by validation_step too!
        return loss

# This is it! It should do full aleatoric + epistemic UQ with VI
# Verified that forward parametrize-caching is redundant here 10/8/24
class PPOU_NetSimulator(POU_NetSimulator, PPOU_net):
    Sim=_UQ_Sim # Sim class for this class (e.g. Sim or Sim_UQ)

if __name__=='__main__':
    # sets up simulation...

    # number of grid points
    nx = ny = nz = 256
    #length of domain
    Lx = Ly = Lz = 2*np.pi
    # viscosity
    nu = 0.003
    # timestep
    dt = 1e-5
    sim = _Sim(nx,ny,nz,Lx,Ly,Lz,nu,dt)

    # generate initial condition (IC)
    u0 = sim.genIC()
    print(f'{u0.shape=}')
    plt.imshow(u0[0,:,:,0]);plt.colorbar()

    # evolve by IC by 4 timesteps
    evolved = sim.evolve(u0,4)
    print(f'{evolved.shape=}')
    print(f'{evolved[0,:,:,0].shape=}')
    plt.imshow(evolved[0,:,:,0]);plt.colorbar()