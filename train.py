#!/usr/bin/env python
# coding: utf-8

import os, sys
import torch

k_modes=eval(str(os.environ.get('K_MODES', None))) # None=maximum (e.g. [103,26,77]) can be a list, GOTCHA: don't change!
assert k_modes is None or type(k_modes) in [int, list, tuple]
stride=eval(str(os.environ.get('STRIDE', 1))) # strides data and k_modes if k_modes=None
assert type(stride) in [int, float, list, tuple]
n_experts: int=int(os.environ.get('N_EXPERTS', 3)) # number of experts in MoE
n_layers: int=int(os.environ.get('N_LAYERS', 4)) # number of layers in the POU net
n_filters: int=int(os.environ.get('N_FILTERS', 32)) # hidden layer width (aka # of filters)
time_chunking: int=int(os.environ.get('TIME_CHUNKING', 9)) # how many self-aware recursive steps to take
time_stride: int=int(os.environ.get('TIME_STRIDE', 1)) # temporal stride between selected frames
batch_size: int=int(os.environ.get('BATCH_SIZE', 2)) # batch size, with VI experts we can only fit 1 batch w/ 20 A100
scale_lr=True # multiply by DDP (total) batch_size
lr: float=float(os.environ.get('LR', 1.563e-5)) # (VI) learning rate (will be scaled by recurisve steps)
weight_decay: float=float(os.environ.get('WEIGHT_DECAY', 0.0)) # weight decay
max_epochs=int(os.environ.get('MAX_EPOCHS', 500))
gradient_clip_val=float(os.environ.get('GRAD_CLIP', 50)) # grad clip adjusted based on new scaling rule
make_optim=eval(f"torch.optim.{os.environ.get('OPTIM', 'Adam')}")
ckpt_path=os.environ.get('CKPT_PATH', None)

use_proportional_k_size=bool(int(os.environ.get('MAKE_K_SIZE_PROPORTIONAL', False))) # if K_MODES or CNN_FILTER_SIZE is given as a integer will create a list where each dimension is proportional to the field size
use_PDE_solver=bool(int(os.environ.get('USE_PDE_SOLVER', True))) # whether to use the PDE solver
use_normalized_MoE=bool(int(os.environ.get('USE_NORMALIZED_MOE', True)))
use_CNN_experts=bool(int(os.environ.get('USE_CNN_EXPERTS', False)))
use_WNO3d_experts=bool(int(os.environ.get('USE_WNO3D_EXPERTS', False))) # whether to use WNO3d as experts
use_IUFNO_experts=bool(int(os.environ.get('USE_IUFNO_EXPERTS', False))) # IUFNO experts
WNO3d_level=int(os.environ.get('WNO3D_LEVEL', 1)) # wavelet decomposition level for WNO3d (default 2)
CNN_filter_size=eval(str(os.environ.get('CNN_FILTER_SIZE', 6))) # only used if use_CNN_experts=True
assert type(CNN_filter_size) in [int, list, tuple]

# Validate that only one expert type is selected
expert_types = [use_CNN_experts, use_WNO3d_experts, use_IUFNO_experts]
assert sum(expert_types) <= 1, f"Only one expert type can be selected. Currently selected: CNN={use_CNN_experts}, WNO3d={use_WNO3d_experts}, IUFNO={use_IUFNO_experts}"

use_fast_dataloaders = bool(int(os.environ.get('FAST_DATALOADERS', False))) # marginally faster dataloaders which use more VRAM
use_trig = bool(int(os.environ.get('TRIG_ENCODINGS', True))) # Ravi's trig encodings
use_grid_inputs = bool(int(os.environ.get('GRID_INPUTS', 0))) # append positional encodings to inputs
out_norm_groups = int(os.environ.get('OUT_NORM_GROUPS', 1)) # 0 or 1 or maybe 2 (whether or not to use output layer norm) keep it at one generally
hidden_norm_groups = int(os.environ.get('HIDDEN_NORM_GROUPS', 1))

# Bayesian settings
use_VI = bool(int(os.environ.get('VI', True))) # whether to enable VI
VI_counts_timestride_gap_data = bool(int(os.environ.get('VI_COUNTS_TIMESTRIDE_GAP_DATA', True))) # whether to count the gap data between timestrides in the dataset size for VI
VI_prior_sigma=float(os.environ.get('VI_PRIOR_SIGMA', 0.2)) # this prior sigma almost matches he sigma of initialization

# NOTE: cosine+warmrestarts is the default scheduler (with T_max=1)
one_cycle=bool(int(os.environ.get('ONE_CYCLE', True))) # scheduler
three_phase=bool(int(os.environ.get('THREE_PHASE', False))) # adds decay after inital bump
RLoP=bool(int(os.environ.get('RLoP', False))) # scheduler
RLoP_factor=0.9
RLoP_patience=15

## original defaults:
# TIME_CHUNKING=9 or 10
# GRAD_CLIP=2.5e-3 --> 2.5e-3*(3 or sqrt(8))=7.5e-3 or 7.071e-3
# VI_LR=1.25e-4 --> 1.25e-4*20=2.5e-3
# batch_size=20 (=1x20 nodes)

## new values:
# TIME_CHUNKING=9
# GRAD_CLIP=(2.5e-3*sqrt(8))*sqrt((9-1)*20)=2.5e-3*16*sqrt(5)=8.944e-2
# VI_LR=1.25e-4*20/(8*20)=1.25e-4/8=1.563e-5
# batch_size=20 (=1x20 nodes)

# Import External Libraries
import torch
import pytorch_lightning as L
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# the flags above make roughly 14% of all ops use tensor cores!

# Import Custom Modules
from MOR_Operator import MOR_Operator
from lightning_utils import *
from POU_net import FieldGatingNet, EqualizedFieldGatingNet
from JHTDB_sim_op import PPOU_NetSimulator, POU_NetSimulator
from JHTDB_data_loading import JHTDBDataModule
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
L.seed_everything(int(os.environ.get('SEED', 0)))

def proportional_allocation(scalar_allocation, proportional_to_size, int_cast=True, cap_at_prop_size=False):
    """
    Dynamically allocates a quantity across dimensions proportionally to the size of those dimensions.
    Such that prod(new_size)==scalar_allocation**len(proportional_to_size)

    This is equivalent to resizing a hyper-cube with side_length=scalar_allocation,
    to be proportional to proportional_to_size while retaining the same volume.
    """
    c=((scalar_allocation**len(proportional_to_size))/np.prod(proportional_to_size))**(1/len(proportional_to_size))
    new_size = np.asarray(proportional_to_size)*c
    if cap_at_prop_size: new_size = np.minimum(new_size, proportional_to_size)
    if int_cast: new_size = (new_size+0.5).astype(int)
    print(f'{new_size=}')
    return tuple(new_size)

if __name__=='__main__':
    # setup data module
    dm = JHTDBDataModule(dataset_path='data/turbulence_output',
                         batch_size=batch_size,
                         time_chunking=time_chunking,
                         stride=stride,
                         time_stride=time_stride,
                         fast_dataloaders=use_fast_dataloaders)

    # derive field size from data module
    field_size = dm.field_size
    if k_modes is None: # default=max (potentially adjusted for stride)
        k_modes=field_size # e.g. [103,26,77]
        assert len(k_modes)==3
    if use_proportional_k_size and type(k_modes) is int:
        k_modes=proportional_allocation(k_modes, field_size, cap_at_prop_size=True)

    ndims=3
    num_nodes = int(os.environ.get('SLURM_STEP_NUM_NODES', 1)) # can be auto-detected by slurm
    num_gpus_per_node = int(os.environ.get('SLURM_STEP_TASKS_PER_NODE', torch.cuda.device_count()))
    print(f'{num_nodes=}, {num_gpus_per_node=}')

    SimModelClass, optional_kwd_args = POU_NetSimulator, {}
    if use_VI: # VI is optional
        if weight_decay > 0: raise ValueError("Weight decay is not supported for VI (use prior_sigma instead!)")

        SimModelClass = PPOU_NetSimulator
        dataset_size_divisor = 1 if VI_counts_timestride_gap_data else time_stride
        dataset_size = model_agnostic_BNN.get_dataset_size(dm.train_dataset)//dataset_size_divisor
        optional_kwd_args = {'prior_cfg': {'prior_sigma': VI_prior_sigma}, 'train_dataset_size': dataset_size}

    # scale lr & grad clip by: the number of *output* timesteps in one full batch (this follows scaling equations)
    scale_of_batch_data = num_nodes*num_gpus_per_node*batch_size*(time_chunking-1) # (includes time)
    print(f'b4 scaling: {lr=}, {gradient_clip_val=}')
    if scale_lr: lr *= scale_of_batch_data
    gradient_clip_val /= scale_of_batch_data**0.5
    print(f'after scaling: {lr=}, {gradient_clip_val=}')
    print(f'{scale_of_batch_data=}')

    if ckpt_path: # secretly use the load from checkpoint api if needed
        SimModelClass_ = SimModelClass
        SimModelClass = lambda **kwd_args: SimModelClass_.load_from_checkpoint(ckpt_path, **kwd_args)

    if use_CNN_experts:
        # GOTCHA: can't do this globally because it breaks the IUFNO expert!
        if use_proportional_k_size and type(CNN_filter_size) is int:
            CNN_filter_size=proportional_allocation(CNN_filter_size, field_size, cap_at_prop_size=True)

        # output_norm_groups=1 by default but specified by the user
        optional_kwd_args |= {'make_expert': CNN, 'k_size': CNN_filter_size, 'skip_connections': True}
    elif use_WNO3d_experts:
        sys.path.append(f'{os.getcwd()}/WNO/Version_2.0.0')
        from wno3d_NS import WNO3d

        # POU_net calls: make_expert(n_inputs, n_outputs, ndims=ndims, **kwd_args)
        # WNO3d needs: (in_channels, out_channels, size, **other_args)
        optional_kwd_args |= {'make_expert': WNO3d, 'size': field_size, 'level': WNO3d_level}
    elif use_IUFNO_experts:
        sys.path.append(f'{os.getcwd()}/IUFNO-CHL')
        from IUFNO import IUFNO4d # GOTCHA: k_size is actually the kernel size for the U-net! and must be an integer
        optional_kwd_args |= {'make_expert': IUFNO4d, 'k_modes': k_modes, 'k_size': CNN_filter_size}
    else: optional_kwd_args['k_modes']=k_modes # assuming MOR_Operator expert

    # NOTE: we need to update field size based on the stride
    simulator_kwd_args = {'nx': field_size[0], 'ny': field_size[1], 'nz': field_size[2], 'dt': 0.0065*time_stride, 'use_PDE_solver': use_PDE_solver}
    make_gating_net = EqualizedFieldGatingNet if use_normalized_MoE else FieldGatingNet
    model = SimModelClass(n_inputs=ndims, n_outputs=ndims, ndims=ndims, n_experts=n_experts, n_layers=n_layers, hidden_channels=n_filters, n_steps=time_chunking-1,
                          weight_decay=weight_decay, lr=lr, one_cycle=one_cycle, three_phase=three_phase, RLoP=RLoP, RLoP_factor=RLoP_factor, RLoP_patience=RLoP_patience,
                          trig_encodings=use_trig, grid_inputs=use_grid_inputs, hidden_norm_groups=hidden_norm_groups, out_norm_groups=out_norm_groups,
                          make_optim=make_optim, make_gating_net=make_gating_net, simulator_kwd_args=simulator_kwd_args, **optional_kwd_args)

    print(f'num model parameters: {utils.count_parameters(model):.5e}')
    print('model:')
    print(model)

    import os, wandb, signal
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.plugins.environments import SLURMEnvironment
    # SLURMEnvironment plugin enables auto-requeue

    job_name = os.environ.get("SLURM_JOB_NAME", 'JHTDB_MOR_MoE')
    version = os.environ.get("SLURM_JOB_ID", None)
    wandb.login(key='251c77a548925cf7f08eecaf2b159ea8d49457c3')
    logger = WandbLogger(project="MOR_MoE", name=job_name, version=version)#, log_model="all")
    #logger.watch(model) # For W&B to log gradients and model topology
    #logger.experiment.config.update({'grad_clip': gradient_clip_val, 'VI_prior_sigma': VI_prior_sigma})

    # Weight-only sharded checkpoints are needed to avoid OOM problem caused by large model size
    model_checkpoint_callback=L.callbacks.ModelCheckpoint(f"lightning_logs/{job_name}/{version}", save_weights_only=True, save_last=False,
                                                          monitor='val_loss/dataloader_idx_0', auto_insert_metric_name=True) # monitor long-horizon loss
    #model_checkpoint_callback=L.callbacks.ModelCheckpoint(f"lightning_logs/{job_name}/{version}", every_n_epochs=100) # simpler resumable checkpointing

    # train model
    strategy = L.strategies.FSDPStrategy(state_dict_type='sharded') if num_nodes*num_gpus_per_node > 1 else 'auto' # sharded reduces peak memory usage but still allows resuming in full!
    trainer = L.Trainer(max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm='value',
                        accelerator='gpu', strategy=strategy, num_nodes=num_nodes, devices=num_gpus_per_node,
                        profiler='simple', logger=logger, plugins=[SLURMEnvironment()], log_every_n_steps=20,
                        callbacks=[model_checkpoint_callback, MemMonitorCallback()])

    # long validation loader causes various problems with profiler & GPU utilization...
    trainer.fit(model, datamodule=dm)#, ckpt_path=ckpt_path)

    # save last checkpoint (doing so manually avoids constant resaving after every epoch)
    trainer.save_checkpoint(f"lightning_logs/{job_name}/{version}/last.ckpt", weights_only=True)
