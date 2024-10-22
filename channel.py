#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.optim import lr_scheduler

k_modes=[103,26,77] # can be a list
n_experts: int=2 # number of experts in MoE
time_chunking: int=9 # how many self-aware recursive steps to take
batch_size: int=1 # batch size, with VI experts we can only fit 1 batch on 20 GPUs!
scale_lr=True # scale with DDP batch_size
lr: float=float(os.environ.get('LR', 1.25e-4)) # learning rate
max_epochs=int(os.environ.get('MAX_EPOCHS', 500))
gradient_clip_val=float(os.environ.get('GRAD_CLIP', 5e-3))
make_optim=eval(f"torch.optim.{os.environ.get('OPTIM', 'Adam')}")
ckpt_path=os.environ.get('CKPT_PATH', None)

use_VI = bool(int(os.environ.get('VI', True))) # whether to enable VI
prior_sigma=float(os.environ.get('PRIOR_SIGMA', 0.2)) # this prior sigma almost matches he sigma of initialization
T_max: int=1 # T_0 for CosAnnealing+WarmRestarts
one_cycle=bool(int(os.environ.get('ONE_CYCLE', True))) # scheduler
three_phase=bool(int(os.environ.get('THREE_PHASE', False))) # adds decay after inital bump
RLoP=bool(int(os.environ.get('RLoP', False))) # scheduler
RLoP_factor=0.9
RLoP_patience=15

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

## wrapper to nullify the VI kwd_args (for compatibility)
#class POU_NetSimulator(POU_NetSimulator):
#    def __init__(self, *args, prior_cfg={}, train_dataset_size=None, **kwd_args):
#        super().__init__(*args, **kwd_args)

if __name__=='__main__':
    # setup dataset
    long_horizon_multiplier=10
    dataset = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking)
    dataset_long_horizon = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking*long_horizon_multiplier)
    _, val_long_horizon = torch.utils.data.random_split(dataset_long_horizon, [0.5, 0.5]) # ensure there are two validation steps
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*long_horizon_multiplier, num_workers=8)
    val_long_loader = torch.utils.data.DataLoader(val_long_horizon, batch_size=batch_size, num_workers=8)
    print(f'{len(dataset)=}\n{len(train_loader)=}\n{len(val_dataset)=}')

    IC_0, Sol_0 = dataset[0]
    print(f'{IC_0.shape=}\n{Sol_0.shape=}')

    ndims=3
    num_nodes = int(os.environ.get('SLURM_STEP_NUM_NODES', 1)) # can be auto-detected by slurm
    print(f'{num_nodes=}')

    SimModelClass, VI_kwd_args = POU_NetSimulator, {}
    if use_VI: # VI is optional
        SimModelClass = PPOU_NetSimulator
        VI_kwd_args = {'prior_cfg': {'prior_sigma': prior_sigma}, 'train_dataset_size': model_agnostic_BNN.get_dataset_size(train_dataset)}
    if ckpt_path: # secretly use the load from checkpoint api if needed
        SimModelClass_ = SimModelClass
        SimModelClass = lambda **kwd_args: SimModelClass_.load_from_checkpoint(ckpt_path, **kwd_args)

    # train model
    if scale_lr: lr *= num_nodes
    model = SimModelClass(n_inputs=ndims, n_outputs=ndims, n_experts=n_experts, ndims=ndims, lr=lr, make_optim=make_optim, T_max=T_max,
                          one_cycle=one_cycle, three_phase=three_phase, RLoP=RLoP, RLoP_factor=RLoP_factor, RLoP_patience=RLoP_patience,
                          n_steps=time_chunking-1, k_modes=k_modes, trig_encodings=True, **VI_kwd_args) #prior_cfg={'prior_sigma': prior_sigma},
                          #train_dataset_size=model_agnostic_BNN.get_dataset_size(train_dataset))

    import os, signal
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    # SLURMEnvironment plugin enables auto-requeue

    job_name = os.environ.get("SLURM_JOB_NAME", 'JHTDB_MOR_MoE')
    version = os.environ.get("SLURM_JOB_ID", None) if job_name!='interactive' else None
    logger = TensorBoardLogger("lightning_logs", name=job_name, version=version)
    profiler = L.profilers.PyTorchProfiler(profile_memory=True, with_stack=True,
                                           on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir))
                                           #schedule=torch.profiler.schedule(skip_first=10, wait=5, warmup=2, active=6, repeat=3))

    # This is needed to avoid problem caused by large model size
    model_checkpoint_callback=L.callbacks.ModelCheckpoint(save_weights_only=True, monitor='loss')
    strategy = L.strategies.FSDPStrategy(state_dict_type='sharded')
    trainer = L.Trainer(max_epochs=max_epochs, accelerator='gpu', strategy=strategy, num_nodes=num_nodes,
                        gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='value', #detect_anomaly=True,
                        profiler=profiler, logger=logger, plugins=[SLURMEnvironment()], callbacks=[model_checkpoint_callback])

    val_dataloaders = [val_loader, val_long_loader] # long validation loader causes various problems with profiler & GPU utilization...
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_dataloaders) #, ckpt_path=ckpt_path)
    #trainer.validate(model=model, dataloaders=val_dataloaders) #, ckpt_path=ckpt_path)
