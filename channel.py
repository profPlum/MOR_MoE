#!/usr/bin/env python
# coding: utf-8

import os
import torch

k_modes=[103,26,77] # can be a list, GOTCHA: don't change!
n_experts: int=int(os.environ.get('N_EXPERTS', 3)) # number of experts in MoE
n_layers: int=int(os.environ.get('N_LAYERS', 4)) # number of layers in the POU net
n_filters: int=int(os.environ.get('N_FILTERS', 32)) # hidden layer width (aka # of filters)
time_chunking: int=int(os.environ.get('TIME_CHUNKING', 9)) # how many self-aware recursive steps to take
batch_size: int=int(os.environ.get('BATCH_SIZE', 2)) # batch size, with VI experts we can only fit 1 batch w/ 20 A100
scale_lr=True # multiply by DDP (total) batch_size
lr: float=float(os.environ.get('LR', 1.25e-4)) # (VI) learning rate (will be scaled by recurisve steps)
max_epochs=int(os.environ.get('MAX_EPOCHS', 500))
gradient_clip_val=float(os.environ.get('GRAD_CLIP', 2.5e-3)) # grad clip adjusted based on new scaling rule
make_optim=eval(f"torch.optim.{os.environ.get('OPTIM', 'Adam')}")
ckpt_path=os.environ.get('CKPT_PATH', None)

use_total_variance=bool(int(os.environ.get('TOTAL_VARIANCE', False)))
use_trig = bool(int(os.environ.get('TRIG_ENCODINGS', True))) # Ravi's trig encodings
use_VI = bool(int(os.environ.get('VI', True))) # whether to enable VI
prior_sigma=float(os.environ.get('PRIOR_SIGMA', 0.2)) # this prior sigma almost matches he sigma of initialization
T_max: int=1 # T_0 for CosAnnealing+WarmRestarts
one_cycle=bool(int(os.environ.get('ONE_CYCLE', True))) # scheduler
three_phase=bool(int(os.environ.get('THREE_PHASE', False))) # adds decay after inital bump
RLoP=bool(int(os.environ.get('RLoP', False))) # scheduler
RLoP_factor=0.9
RLoP_patience=15

# Import External Libraries
import torch
import pytorch_lightning as L
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# the flags above make roughly 14% of all ops use tensor cores!

# Import Custom Modules

from lightning_utils import *
from POU_net import POU_net, PPOU_net, FieldGatingNet, EqualizedFieldGatingNet
from JHTDB_sim_op import PPOU_NetSimulator, POU_NetSimulator, JHTDB_Channel
import model_agnostic_BNN
import utils

class MemMonitorCallback(L.Callback):
    def __init__(self, clear_interval=40):
        self._epoch_counter=0
        self._clear_interval=clear_interval
    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_counter+=1
        utils.nvidia_smi(clear_mem=not (self._epoch_counter%self._clear_interval), verbose=True)
        print(f'SLURM_LOCALID={os.environ["SLURM_LOCALID"]}, GPU_Id={torch.cuda.current_device()}', flush=True)

# for ablation study
L.seed_everything(0)

if __name__=='__main__':
    # setup dataset
    long_horizon_multiplier=10 # longer evaluation time window is X times the shorter training time window (can e.g. detect NaNs)
    dataset = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking) # called dataloader_idx_0 in tensorboard
    dataset_long_horizon = JHTDB_Channel('data/turbulence_output', time_chunking=time_chunking*long_horizon_multiplier) # called dataloader_idx_1 in tensorboard
    _, val_long_horizon = torch.utils.data.random_split(dataset_long_horizon, [0.5, 0.5]) # 50% ensures there are two validation steps
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=32, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*long_horizon_multiplier, num_workers=16)
    val_long_loader = torch.utils.data.DataLoader(val_long_horizon, batch_size=batch_size, num_workers=16)
    print(f'{len(dataset)=}\n{len(train_loader)=}\n{len(val_dataset)=}')

    IC_0, Sol_0 = dataset[0]
    print(f'{IC_0.shape=}\n{Sol_0.shape=}')

    ndims=3
    num_nodes = int(os.environ.get('SLURM_STEP_NUM_NODES', 1)) # can be auto-detected by slurm
    num_gpus_per_node = int(os.environ.get('SLURM_STEP_TASKS_PER_NODE', torch.cuda.device_count()))
    print(f'{num_nodes=}, {num_gpus_per_node=}')

    SimModelClass, VI_kwd_args = POU_NetSimulator, {}
    if use_VI: # VI is optional
        SimModelClass = PPOU_NetSimulator
        VI_kwd_args = {'prior_cfg': {'prior_sigma': prior_sigma}, 'train_dataset_size': model_agnostic_BNN.get_dataset_size(train_dataset),
                       'total_variance': use_total_variance}
    if ckpt_path: # secretly use the load from checkpoint api if needed
        SimModelClass_ = SimModelClass
        SimModelClass = lambda **kwd_args: SimModelClass_.load_from_checkpoint(ckpt_path, **kwd_args)

    # scale lr & grad clip
    if scale_lr: lr *= num_nodes*num_gpus_per_node*batch_size
    gradient_clip_val *= (time_chunking-1)**0.5

    # train model
    model = SimModelClass(n_inputs=ndims, n_outputs=ndims, ndims=ndims, n_experts=n_experts, n_layers=n_layers, hidden_channels=n_filters, make_optim=make_optim,
                          lr=lr, T_max=T_max, one_cycle=one_cycle, three_phase=three_phase, RLoP=RLoP, RLoP_factor=RLoP_factor, RLoP_patience=RLoP_patience,
                          n_steps=time_chunking-1, k_modes=k_modes, trig_encodings=use_trig, make_gating_net=EqualizedFieldGatingNet, **VI_kwd_args)

    print(f'num model parameters: {utils.count_parameters(model):.2e}')
    print('model:')
    print(model)

    import os, wandb, signal
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    # SLURMEnvironment plugin enables auto-requeue

    job_name = os.environ.get("SLURM_JOB_NAME", 'JHTDB_MOR_MoE')
    version = os.environ.get("SLURM_JOB_ID", None)
    wandb.login(key='251c77a548925cf7f08eecaf2b159ea8d49457c3')
    logger = WandbLogger(project="MOR_MoE", log_model="all", name=job_name, version=version)
    logger.watch(model) # For W&B to log gradients and model topology

    # Weight-only sharded checkpoints are needed to avoid problem caused by large model size
    model_checkpoint_callback=L.callbacks.ModelCheckpoint(f"lightning_logs/{job_name}/{version}", save_weights_only=True,
                                                          monitor='val_loss/dataloader_idx_1') # monitor long-horizon loss
    strategy = L.strategies.FSDPStrategy(state_dict_type='sharded')
    trainer = L.Trainer(max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='value',
                        accelerator='gpu', strategy=strategy, num_nodes=num_nodes, devices=num_gpus_per_node,
                        profiler='simple', logger=logger, plugins=[SLURMEnvironment()], log_every_n_steps=20,
                        callbacks=[model_checkpoint_callback, MemMonitorCallback()]) #detect_anomaly=True,

    val_dataloaders = [val_loader, val_long_loader] # long validation loader causes various problems with profiler & GPU utilization...
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_dataloaders) #, ckpt_path=ckpt_path)
    #trainer.validate(model=model, dataloaders=val_dataloaders) #, ckpt_path=ckpt_path)
