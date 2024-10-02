#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.optim import lr_scheduler

k_modes=[103,26,77] # can be a list
n_experts: int=2 # number of experts in MoE
time_chunking: int=5 # how many self-aware recursive steps to take
batch_size: int=1 # batch size, with VI experts we can only fit 1 batch on 20 GPUs!
scale_lr=True # scale with DDP batch_size
lr: float=float(os.environ.get('LR', 0.00025)) # learning rate
max_epochs=int(os.environ.get('MAX_EPOCHS', 500))
gradient_clip_val=float(os.environ.get('GRAD_CLIP', 0.5))
ckpt_path=os.environ.get('CKPT_PATH', None)
make_optim=eval(f"torch.optim.{os.environ.get('OPTIM', 'Adam')}")

prior_sigma=float(os.environ.get('PRIOR_SIGMA', 1.0))
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
#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# the flags above make roughly 14% of all ops use tensor cores!

# Import Custom Modules

from lightning_utils import *
from MOR_Operator import MOR_Operator, MOR_Layer
from POU_net import POU_net, PPOU_net, FieldGatingNet
from JHTDB_sim_op import PPOU_NetSimulator, POU_NetSimulator, Sim, JHTDB_Channel
import model_agnostic_BNN
import utils

class MemMonitorCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        utils.report_cuda_memory_usage(clear=False)
    def on_validation_epoch_end(self, trainer, pl_module):
        utils.report_cuda_memory_usage(clear=False)

if __name__=='__main__':
    # setup dataset
    dataset = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking)
    dataset_long_horizon = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking*2)
    _, val_long_horizon = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    #val_long_loader = torch.utils.data.DataLoader(val_long_horizon, batch_size=batch_size, num_workers=8)
    print(f'{len(dataset)=}\n{len(train_loader)=}\n{len(val_dataset)=}')

    IC_0, Sol_0 = dataset[0]
    print(f'{IC_0.shape=}\n{Sol_0.shape=}')

    ndims=3
    num_nodes = int(os.environ.get('SLURM_STEP_NUM_NODES', 1)) # can be auto-detected by slurm
    print(f'{num_nodes=}')

    # train model
    if scale_lr: lr *= num_nodes
    model = PPOU_NetSimulator(ndims, ndims, n_experts, ndims=ndims, lr=lr, make_optim=make_optim, T_max=T_max, prior_cfg={'prior_sigma': prior_sigma},
                              one_cycle=one_cycle, three_phase=three_phase, RLoP=RLoP, RLoP_factor=RLoP_factor, RLoP_patience=RLoP_patience,
                              n_steps=time_chunking-1, k_modes=k_modes)

    import os, signal
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    # SLURMEnvironment plugin enables auto-requeue

    logger = TensorBoardLogger("lightning_logs", name=os.environ.get("SLURM_JOB_NAME", 'JHTDB_MOR_MoE'),
                                version=os.environ.get("SLURM_JOB_ID", None))
    profiler = L.profilers.PyTorchProfiler(profile_memory=True, with_stack=True,
                                           on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir))
                                           #schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=0))
                                           ## TODO: remove when done debugging

    # This is needed to avoid problem caused by large model size
    model_checkpoint_callback=L.callbacks.ModelCheckpoint(save_weights_only=True)
    strategy = L.strategies.FSDPStrategy(state_dict_type='sharded')
    trainer = L.Trainer(max_epochs=max_epochs, accelerator='gpu', strategy=strategy, num_nodes=num_nodes,
                        gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='value', # regularization isn't good for OneCycleLR
                        profiler=profiler, logger=logger, plugins=[SLURMEnvironment()], callbacks=[model_checkpoint_callback])

    val_dataloaders = [val_loader] #, val_long_loader] # long validation loader causes various problems with profiler & GPU utilization...
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_dataloaders, ckpt_path=ckpt_path)
