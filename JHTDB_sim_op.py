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
    def __init__(self,nx=103,ny=26,nz=77,Lx=8*np.pi,Ly=2.0,Lz=3*np.pi,nu=5e-5,dt=0.0065,
                 u_b: torch.Tensor=0, use_PDE_solver=True, anisotropic_filter: bool=False, disable_filter: bool=False,
                 dealias_before_quadratic: bool=False, apply_pde_filter_bottleneck: bool=True):
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
        super().__init__()
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
        """Build Sim from data module. nx,ny,nz,dt are taken from the module; all kwd_args are forwarded to the constructor."""
        field_size = data_module.field_size
        merged = {'nx': field_size[0], 'ny': field_size[1], 'nz': field_size[2], 'dt': 0.0065*data_module.time_stride, **kwd_args}
        if use_manual_advection: merged['u_b'] = data_module.u_b
        return cls(**merged)

    def genIC(self, from_LES=False):
        h = torch.tensor(np.random.normal(0,1,(self.nx,self.ny,self.nz,3))).float().to(self.device)
        hh = _rfft(h) * self.filt2[...,None]
        proj = self.k*(torch.sum(self.k*hh,axis=-1)/self.knorm2)[...,None]
        proj[0]=0
        u0 = _irfft(hh - proj, s=self.shapef)
        u0 = u0.permute(-1,0,1,2) # (i.e. torch.moveaxis(u0,-1,0))
        if from_LES:
            assert self.op is not IdentityOp, 'Cannot use LES IC with IdentityOp'
            with torch.inference_mode():
                u0 = self.evolve(u0,n=20) # make it more realistic (assuming forcing function)
        return u0

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
        u = u0
        outputs = []
        if len(u.shape)==4: # all permute ops above assume 4 dims (before vmap)
            u = u[None] # add batch dim
        for i in range(n):
            u = self.op.forward(self.vmap_NSupd(u)) # NOTE: this is the only place where the operator is used
            if u.isnan().any():
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            #assert not u.isnan().any()
            if intermediate_outputs and i%intermediate_output_stride==0:
                outputs.append(u.to('cpu', non_blocking=True) if to_cpu else u)

        torch.cuda.synchronize() # async sync

        # time dim is the last dim (if it exists)
        outputs = torch.stack(outputs,axis=-1) if intermediate_outputs else u
        return outputs.squeeze() if len(u.shape)>len(u0.shape) else outputs
        # remove artificial batch dimension only if it was added

# For use with PPOU_net
class _UQ_Sim(_Sim):
    def __init__(self, *args, propagate_uq: bool=True, **kwd_args):
        super().__init__(*args, **kwd_args)
        self.propagate_uq = propagate_uq

    def genIC(self, from_LES=False):
        u0 = super(_UQ_Sim,self).genIC(from_LES=from_LES)
        if from_LES: u0 = u0[0] # remove unnecessary uq tensor
        return u0

    # This needs to output intermediate time-steps to get full loss!
    def evolve(self,u0,n,intermediate_outputs=False, intermediate_output_stride=1, to_cpu=False):
        u = u0
        u_outputs = []
        uq_outputs = []
        if len(u.shape)==4: # all permute ops above assume 4 dims (before vmap)
            u = u[None] # add batch dim

        #uq = None
        uq = zero_uq = torch.zeros(1,device=u.device, dtype=u.dtype).expand(*u.shape)
        for i in range(n):
            if not self.propagate_uq: uq = zero_uq
            u, uq = self.op.forward(self.vmap_NSupd(u), uq)
            if u.isnan().any() or uq.isnan().any():
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            #assert not (u.isnan().any() or uq.isnan().any())
            if intermediate_outputs and i%intermediate_output_stride==0:
                u_outputs.append(u.to('cpu', non_blocking=True) if to_cpu else u)
                uq_outputs.append(uq.to('cpu', non_blocking=True) if to_cpu else uq)

        # remove artificial batch dimension only if it was added
        maybe_squeeze = lambda output: output.squeeze() if len(u.shape)>len(u0.shape) else output

        if intermediate_outputs and to_cpu:
            torch.cuda.synchronize() # async sync

        if intermediate_outputs: # time dim is the last dim (if it exists)
            u_outputs = maybe_squeeze(torch.stack(u_outputs,axis=-1))
            uq_outputs = maybe_squeeze(torch.stack(uq_outputs,axis=-1))
            return u_outputs, uq_outputs
        else: return maybe_squeeze(u), maybe_squeeze(uq)

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