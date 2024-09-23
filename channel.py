#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.optim import lr_scheduler

k_modes=[103,26,77] # can be a list
n_experts: int=2 # number of experts in MoE
time_chunking: int=5 # how many self-aware recursive steps to take
batch_size: int=2 # batch size
scale_lr=True # scale with DDP batch_size
lr: float=float(os.environ.get('LR', 0.0005)) # learning rate
max_epochs=int(os.environ.get('MAX_EPOCHS', 500))
gradient_clip_val=float(os.environ.get('GRAD_CLIP', 0.5))
ckpt_path=os.environ.get('CKPT_PATH', None)
make_optim=eval(f"torch.optim.{os.environ.get('OPTIM', 'Adam')}")

T_max: int=1 # T_0 for CosAnnealing+WarmRestarts
one_cycle=bool(int(os.environ.get('ONE_CYCLE', False))) # scheduler
three_phase=bool(int(os.environ.get('THREE_PHASE', False))) # adds decay after inital bump
RLoP=bool(int(os.environ.get('RLoP', False))) # scheduler
RLoP_factor=0.9
RLoP_patience=25

# Import External Libraries

import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
#torch.set_float32_matmul_precision('medium')
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

# Import Custom Modules

from lightning_utils import *
from MOR_Operator import MOR_Operator
from POU_net import POU_net, FieldGatingNet
from JHTDB_sim_op import POU_NetSimulator, Sim, JHTDB_Channel


if __name__=='__main__':
    # setup dataset
    dataset = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking)
    dataset_long_horizon = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking*3)
    _, val_long_horizon = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    val_long_loader = torch.utils.data.DataLoader(val_long_horizon, batch_size=batch_size, num_workers=8)
    print(f'{len(dataset)=}\n{len(train_loader)=}\n{len(val_dataset)=}')

    IC_0, Sol_0 = dataset[0]
    print(f'{IC_0.shape=}\n{Sol_0.shape=}')

    ndims=3
    extra_args = {'k_modes': [103,26,77], 'k': 2, 'n_layers': 4}
    gating_net = lambda *args, **kwd_args: FieldGatingNet(*args, **(kwd_args | extra_args))
    # this lambda does nothing if on RevertingMoESparsity branch...

    num_nodes = int(os.environ.get('SLURM_STEP_NUM_NODES', 1)) # can be auto-detected by slurm
    print(f'{num_nodes=}')

    # train model
    if scale_lr: lr *= num_nodes
    model = POU_NetSimulator(ndims, ndims, n_experts, ndims=ndims, lr=lr, make_optim=make_optim, T_max=T_max, #make_gating_net=gating_net,
                             one_cycle=one_cycle, three_phase=three_phase, RLoP=RLoP, RLoP_factor=RLoP_factor, RLoP_patience=RLoP_patience,
                             n_steps=time_chunking-1, k_modes=k_modes)

    import os, signal
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    # SLURMEnvironment plugin enables auto-requeue

    logger = TensorBoardLogger("lightning_logs", name=os.environ.get("SLURM_JOB_NAME", 'JHTDB_MOR_MoE'),
                                version=os.environ.get("SLURM_JOB_ID", None))
    profiler = L.profilers.PyTorchProfiler(profile_memory=True, with_stack=True)
    trainer = L.Trainer(max_epochs=max_epochs, accelerator='gpu', strategy='fsdp', num_nodes=num_nodes,
                        gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='value', # regularization isn't good for OneCycleLR
                        profiler=profiler, plugins=[SLURMEnvironment()], logger=logger)#, precision='bf16-mixed')#, log_every_n_steps=1)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=[val_loader, val_long_loader], ckpt_path=ckpt_path)
