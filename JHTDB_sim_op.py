import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L

rfft = functools.partial(torch.fft.rfftn,dim=[0,1,2])
irfft = functools.partial(torch.fft.irfftn,dim=[0,1,2])

# only works on *nix
def get_forked_pdb():
    import pdb

    class ForkedPdb(pdb.Pdb):
        """
        A Pdb subclass that may be used
        from a forked multiprocessing child
        """
        def interaction(self, *args, **kwargs):
            _stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin
    return ForkedPdb()


def divide_no_nan(a,b):
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

class Sim(L.LightningModule):
    '''
    Raw Sim[ulator] class that solves naiver stokes with learned model correction.
    We wrapped Dr. Patel's original code to do axis swapping
    (the code needs channel dim last but pytorch needs it right after batch dim),
    in a way that is *compatible with vmap* for batching!!
    '''
    def __init__(self,nx,ny,nz,Lx,Ly,Lz,nu,dt):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nu = nu
        self.k = torch.tensor(np.stack(np.meshgrid(np.fft.fftfreq(nx)*nx*2.*np.pi/Lx,
                                       np.fft.fftfreq(ny)*ny*2.*np.pi/Ly,
                                       np.fft.rfftfreq(nz)*nz*2.*np.pi/Lz,indexing='ij'),axis=-1)).cfloat()

        self.x = torch.tensor(np.stack(np.meshgrid(np.arange(nx)/nx*Lx,
                                       np.arange(ny)/ny*Ly,
                                       np.arange(nz)/nz*Lz,indexing='ij'),axis=-1))

        self.xi = (self.x[...,0]>=np.pi/4)*(self.x[...,0]<=Lx-np.pi/4)

        self.knorm2 = torch.sum(self.k**2,-1).real.float()
        self.Ainv =  torch.tensor(1./(1.+nu*np.einsum('...j,...j->...',self.k,self.k)))
        self.filt = torch.as_tensor((np.sqrt(self.knorm2)<=2./3*(min(self.nx,self.ny,self.nz)/2+1)))
        self.filt2 = torch.as_tensor((np.sqrt(self.knorm2)<=1./3*(min(self.nx,self.ny,self.nz)/2+1)))
        self.Ainv = self.Ainv * self.filt
        self.dt = dt
        self.shapef = [nx,ny,nz]
        self.shapeh = [nx,ny,nz//2+1]
        self.forcing = 0.*self.k
        self.forcing[4,4,4,0] = 10.

        self.eta = 1e-3
        self.nu_num = 1e-3
        self.op = IdentityOp() # identity by default

        for name, value in vars(self).copy().items():
            if isinstance(value, torch.Tensor):
                del vars(self)[name]
                self.register_buffer(name, value)

    def genIC(self):
        h = torch.tensor(np.random.normal(0,1,(self.nx,self.ny,self.nz,3))).float()
        hh = rfft(h) * self.filt2[...,None]
        proj = self.k*(torch.sum(self.k*hh,axis=-1)/self.knorm2)[...,None]
        proj[0]=0
        u0 = irfft(hh - proj, s=self.shapef)
        return u0.permute(-1,0,1,2) # (i.e. torch.moveaxis(u0,-1,0))

    def NSupd(self,u): # Navier-stokes update
        u = u.permute(1,2,-1,0) #torch.moveaxis(u, 0, -1)
        uh = rfft(u)
        assert list(uh.shape)[:-1]==self.shapeh
        u2h = rfft(torch.einsum('...i,...j->...ij',u,u))
        u = irfft(self.Ainv[...,None]*(
            uh + self.dt*(-1.j*torch.einsum('...j,...ij->...i',self.k,u2h)
                 + 1.j*divide_no_nan(torch.einsum('...i,...j,...k,...jk->...i',self.k,self.k,self.k,u2h),self.knorm2[...,None])
                 )),
                 s=self.shapef
            )
        return u.permute(-1,0,1,2) # i.e. torch.moveaxis(u, -1, 0)

    # set the neural operator for correction
    def set_operator(self, op):
        self.op = op

    # This should perhaps use super().forward()
    # use operator learning here to correct for missing physics
    def learnedCorrection(self,u):
        forward = self.op.forward(u)
        #assert torch.isreal(u).all() and torch.isreal(forward).all()
        return forward
        # __call__() is necessary for hooks... <- but this should already happen outside!

    # This needs to output intermediate time-steps to get full loss!
    # GOTCHA: this probably needs torch.vmap to work properly across a batch
    def evolve(self,u0,n,intermediate_outputs=False):
        u = u0
        outputs = []
        for _ in range(n):
            u = self.learnedCorrection(self.NSupd(u)[None]).squeeze()
            outputs.append(u)
        return torch.stack(outputs,axis=-1) if intermediate_outputs else outputs[-1]
        # stacking on the first dimension is correct b/c first dimension is the time dimension!

from lightning_utils import *
from POU_net import POU_net

dump_vmap = lambda func: lambda X: torch.stack([func(x) for x in X])

class POU_NetSimulator(POU_net):
    ''' Combines the POU_net with the raw Sim[ulator] class (internally). '''
    def __init__(self, *args, simulator: Sim, n_steps: int, **kwd_args):
        super().__init__(*args, **kwd_args)
        simulator.set_operator(super()) # this will internally call super().forward(X)
        self.simulator = simulator
        self.n_steps = n_steps # n timesteps for PDE evolution
    def forward(self, X):
        #X.shape==[batch, x, y, ...]
        evolve = lambda u0: self.simulator.evolve(u0, n=self.n_steps, intermediate_outputs=True)
        evolve = torch.vmap(evolve) #torch.vmap(evolve) # vmap across batch dim
        return evolve(X)

import os, sys
import h5py
from glob import glob

# TODO: load & store dataset in one big hdf5 file (more efficient I/O)
# Verified to work: 8/23/24
class JHTDB_Channel(torch.utils.data.Dataset):
    '''
    Dataset for the JHTDB autoregressive problem... It is not possible to make
    this predict everything at once because that would make the dataset size=1.
    '''
    def __init__(self, path:str, time_chunking=5):
        self.time_chunking=time_chunking
        self.path=path
    def __len__(self):
        return len(glob(f'{self.path}/*.h5'))//(self.time_chunking)
    def __getitem__(self, index):
        files = []
        velocity_fields = []
        for i in range(index*self.time_chunking, (index+1)*self.time_chunking):
            i+=1
            try: files.append(h5py.File(f'{self.path}/channel_t={i}.h5', 'r')) # keep open for stacking
            except OSError as e:
                if 'unable to open' in str(e).lower():
                    raise OSError(f'Unable to open file: "{self.path}/channel_t={i}.h5"')
                else: raise
            velocity_fields.append(files[-1][f'Velocity_{i:04}']) # :04 zero pads to 4 digits
        velocity_fields = torch.swapaxes(torch.as_tensor(np.stack(velocity_fields)), 0, -1)
        return velocity_fields[...,0], velocity_fields[...,1:] # X=IC, Y=sol

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
    sim = Sim(nx,ny,nz,Lx,Ly,Lz,nu,dt)

    # generate initial condition (IC)
    u0 = sim.genIC()
    print(f'{u0.shape=}')
    plt.imshow(u0[0,:,:,0]);plt.colorbar()

    # evolve by IC by 4 timesteps
    evolved = sim.evolve(u0,4)
    print(f'{evolved.shape=}')
    print(f'{evolved[0,:,:,0].shape=}')
    plt.imshow(evolved[0,:,:,0]);plt.colorbar()
