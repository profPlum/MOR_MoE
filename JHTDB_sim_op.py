import warnings

import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L

from lightning_utils import *
from POU_net import POU_net, PPOU_net

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
    def __init__(self,nx=103,ny=26,nz=77,Lx=8*np.pi,Ly=2.0,Lz=3*np.pi,nu=5e-5,dt=0.0065, use_PDE_solver=True):
        ''' Defaults are set to the values needed for JHTDB channel flow.
            Also note that nu:=viscosity, Lx,Ly,Lz:=domain dimensions (physical),
            and nx,ny,nz:=grid dimensions (virtual) '''
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nu = nu
        self.use_PDE_solver = use_PDE_solver # whether to use the PDE solver

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
        filt = torch.as_tensor((np.sqrt(self.knorm2)<=2./3*(min(self.nx,self.ny,self.nz)/2+1))) # only used locally
        self.filt2 = torch.as_tensor((np.sqrt(self.knorm2)<=1./3*(min(self.nx,self.ny,self.nz)/2+1)))
        self.Ainv = self.Ainv * filt
        self.dt = dt
        self.shapef = [nx,ny,nz]
        self.shapeh = [nx,ny,nz//2+1]
        #self.forcing = 0.*self.k # not used
        #self.forcing[4,4,4,0] = 10. # not used

        self.eta = 1e-3
        self.nu_num = 1e-3
        self.op = IdentityOp() # identity by default

        for name, value in vars(self).copy().items():
            if isinstance(value, torch.Tensor):
                del vars(self)[name]
                self.register_buffer(name, value.detach(), persistent=False)

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
        u2h = _rfft(torch.einsum('...i,...j->...ij',u,u))
        u = _irfft(self.Ainv[...,None]*(
            uh + self.dt*(-1.j*torch.einsum('...j,...ij->...i',self.k,u2h)
                 + 1.j*_divide_no_nan(torch.einsum('...i,...j,...k,...jk->...i',self.k,self.k,self.k,u2h),self.knorm2[...,None])
                 )),
                 s=self.shapef
            )
        return u.permute(-1,0,1,2) # i.e. torch.moveaxis(u, -1, 0)

    def NSupd(self,u):
        if not self.use_PDE_solver: return u
        return self._NSupd(u)

    # set the neural operator for correction
    def set_operator(self, op):
        self.op = op

    # This needs to output intermediate time-steps to get full loss!
    def evolve(self,u0,n,intermediate_outputs=False, intermediate_output_stride=1):
        u = u0
        outputs = []
        NSupd = torch.vmap(self.NSupd) # only this needs vmapping, NeuralOp is already batched
        if len(u.shape)==4: # all permute ops above assume 4 dims (before vmap)
            u = u[None] # add batch dim
        for i in range(n):
            u = self.op.forward(NSupd(u)) # NOTE: this is the only place where the operator is used
            if u.isnan().any():
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            #assert not u.isnan().any()
            if intermediate_outputs and i%intermediate_output_stride==0: outputs.append(u)

        # time dim is the last dim (if it exists)
        outputs = torch.stack(outputs,axis=-1) if intermediate_outputs else u
        return outputs.squeeze() if len(u.shape)>len(u0.shape) else outputs
        # remove artificial batch dimension only if it was added

# For use with PPOU_net
class _UQ_Sim(_Sim):
    def genIC(self, from_LES=False):
        u0 = super(_UQ_Sim,self).genIC(from_LES=from_LES)
        if from_LES: u0 = u0[0] # remove unnecessary uq tensor
        return u0

    # This needs to output intermediate time-steps to get full loss!
    def evolve(self,u0,n,intermediate_outputs=False, intermediate_output_stride=1):
        u = u0
        u_outputs = []
        uq_outputs = []
        NSupd = torch.vmap(self.NSupd) # only this needs vmapping, NeuralOp is already batched
        if len(u.shape)==4: # all permute ops above assume 4 dims (before vmap)
            u = u[None] # add batch dim
        uq = None
        for i in range(n):
            u, uq = self.op.forward(NSupd(u), uq)
            if u.isnan().any() or uq.isnan().any():
                warnings.warn(f'Simulation has diverged into NaNs! At step: {i}')
            #assert not (u.isnan().any() or uq.isnan().any())
            if intermediate_outputs and i%intermediate_output_stride==0:
                u_outputs.append(u)
                uq_outputs.append(uq)

        # remove artificial batch dimension only if it was added
        maybe_squeeze = lambda output: output.squeeze() if len(u.shape)>len(u0.shape) else output

        if intermediate_outputs: # time dim is the last dim (if it exists)
            return maybe_squeeze(torch.stack(u_outputs,axis=-1)), \
                    maybe_squeeze(torch.stack(uq_outputs,axis=-1))
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
