#!/usr/bin/env python
# coding: utf-8

# In[52]:

time_chunking: int=4 # how many self-aware recursive steps to take
batch_size: int = 2 # batch size
lr: float=0.001 # learning rate
T_max: int=1 # T_0 for CosAnnealing+WarmRestarts
n_experts: int=1 # number of experts in MoE
n_devices: int=6
interactive: bool=False # whether to use parallelism

# In[53]:


# Import External Libraries

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')


# In[54]:


# Import Custom Modules

from lightning_utils import *
from MOR_Operator import MOR_Operator
from POU_net import POU_net

from JHTDB_sim_op import POU_NetSimulator, Sim, JHTDB_Channel

if __name__=='__main__':
        # sets up simulation

    # number of grid points
    #nx = ny = nz = 256
    nx,ny,nz=mesh_shape=np.asarray([77, 26, 103])

    #length of domain
    Lx = Ly = Lz = 2*np.pi
    #Lx,Ly,Lz=2*np.pi*(mesh_shape/256.0) # assume proportional
    # viscosity
    nu = 5e-5
    # timestep
    dt = 0.0013
    sim = Sim(nx,ny,nz,Lx,Ly,Lz,nu,dt)


    # In[57]:


    print(f"{Lx,Ly,Lz=}")


    # In[58]:

    dataset = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    print(f'{len(dataset)=}\n{len(train_loader)=}\n{len(val_dataset)=}')

    IC_0, Sol_0 = dataset[0]
    print(f'{IC_0.shape=}\n{Sol_0.shape=}')

    ndims=3
    Expert = lambda: MOR_Operator(in_channels=ndims, out_channels=ndims, n_layers=4, k_modes = 26, ndims=ndims)
    #Expert = lambda: CNN(ndims=ndims, k_size=3) # works

    # train model
    model = POU_NetSimulator(Expert, n_experts, lr=lr, T_max=T_max,
                             n_inputs=ndims, simulator=sim, n_steps=time_chunking-1)
    kwd_args = {'devices': 1} if interactive else {'strategy': 'fsdp', 'devices': n_devices}
    trainer = L.Trainer(max_epochs=100, accelerator='gpu',
                        gradient_clip_val=1.0, gradient_clip_algorithm='value', **kwd_args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
