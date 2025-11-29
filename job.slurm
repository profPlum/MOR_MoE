#!/bin/bash
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --nodes=1                            # Trying allocating a whole node to play nice with pytorch-lightning
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --mem-per-gpu=115GB                  # CPU? RAM per GPU < 1/4 of the total node RAM = 120GB
#SBATCH --gpus-per-node=1                    # GPUs per task is 1
#SBATCH --cpus-per-task=72                   # CORES per task for data loaders (72 avail per GPU) (very few really used)
#SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS) (48 hours for WNO-600-epochs)
#SBATCH --partition=ghx4                     # Partition name
#SBATCH --account=beoi-dtai-gh
#SBATCH --job-name=MLE_TS=4

# Not going to work anymore b/c we would need to save the entire checkpoint which causes OOM...
# ##SBATCH --signal=SIGUSR1@90
# ##SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS)

# Load modules or environment
. ~/.bashrc
conda activate uqops
#module load cuda/11.8.0

#export NCCL_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=INFO

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun --ntasks-per-node=1 nvidia-smi

## Old values:
#STANDARD_LR=3.125e-5 # theoretically suggested by Adam default lr=0.001/32 s.t. 32 is default batch size
#VI_LR=1.25e-4 # this is 4x higher than theory suggests (it is only possible b/c grad clip)
#MLE_LR=2.5e-4 # this is 8x higher than theory suggests (it is only possible b/c grad clip + determinism)

# GRAD_CLIP=(2.5e-3*sqrt(8))*sqrt((9-1)*20)=2.5e-3*16*sqrt(5)=8.944e-2
# VI_LR=1.25e-4*20/(8*20)=1.25e-4/8=1.563e-5
# these new values recover the old defaults with new scaling rules
VI_LR=1.563e-5
MLE_LR=3.125e-5

# (VI+TRIG) 1550 epochs takes 20 hours (with 4 GH200s, measured 4/2/25)
# Everything 1600 epochs takes 33 hours (with 4 GH200s, measured 9/23/25)
export MAX_EPOCHS=1600 # MUST BE ACCURATELY ADJUSTED based on estimated epochs (this ensures that OneCycle is properly calibrated)
export LR=$MLE_LR # this is 4x higher than theory suggests (it is only possible b/c grad clip)
export GRAD_CLIP=500000 # it is best to keep this value, it will automatically be adjusted based on number of recursive timesteps & batch_size
export WEIGHT_DECAY=0.0 # 0.01 is AdamW default
export OPTIM='AdamW' # RAdam is better for resuming, otherwise use Adam
export RLoP=0 # Reduce Learning-rate on Plateu, can be useful for resuming (but otherwise keep it off!). Also bash booleans are binary
export ONE_CYCLE=$((! RLoP)) # OneCycle warmup & decay lr schedule. Really important for scaling to multiple GPUs!
export THREE_PHASE=0 # faster decay in one cycle, usually not needed
export USE_PDE_SOLVER=1 # whether to use the PDE solver
export TRIG_ENCODINGS=1 # gives better (nearly periodic) partitions
export USE_NORMALIZED_MOE=1 # importance normalization gives better expert utilization
export OUT_NORM_GROUPS=1 # whether to use output normalization (you should!)
export HIDDEN_NORM_GROUPS=1 # whether to use hidden normalization (you should!)
export VI=0 # use VI or not (in bash binary booleans have the best support for boolean logic)
export VI_PRIOR_SIGMA=0.2 # approximately reflects the values of sigma at initialization
export VI_COUNTS_TIMESTRIDE_GAP_DATA=1 # whether to count the gap data between timestrides in the dataset size for VI
export TIME_STRIDE=4 # must be integer-valued
export STRIDE=1 # e.g. 1.5 or a crazy list [1.5,1.3,1.8] for smaller experiments
export TIME_CHUNKING=10 # 10 is default now b/c we can fit it in GH200s
export BATCH_SIZE=2 # GH200 can handle batch_size=2! (it has > x2 the memory of A100)
export FAST_DATALOADERS=0 # warning: they can cause OOMs for production jobs!
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden channels
export N_EXPERTS=3 # number of experts
export MAKE_K_SIZE_PROPORTIONAL=1 # whether to rescale K dims to be proportional to the field size (only works when K_MODES or CNN_FILTER_SIZE is given as an integer)
#export K_MODES=20 # 4 for stride~=1.5, 6 for full STRIDE=1, use this for comparison of FNO to CNN, keep off by default!

# CNN experts?
export USE_CNN_EXPERTS=0
export CNN_FILTER_SIZE=4 # 4 for stride~=1.5, 6 for full STRIDE=1
#export N_LAYERS=11 N_EXPERTS=1

# WNO3d experts?
export USE_WNO3D_EXPERTS=0 # To use WNO3d experts: set USE_WNO3D_EXPERTS=1 and USE_CNN_EXPERTS=0
export WNO3D_LEVEL=1 # wavelet decomposition level (default 2, higher = more detailed wavelet analysis)
#export N_EXPERTS=1 N_LAYERS=4 OUT_NORM_GROUPS=0 HIDDEN_NORM_GROUPS=0
#export STRIDE='[1.609,1.625,1.203]' FAST_DATALOADERS=1 # 64x16x64
#export BATCH_SIZE=2
#export N_FILTERS=36

## IU-FNO experts?
#export USE_IUFNO_EXPERTS=1 CNN_FILTER_SIZE=3 N_EXPERTS=1 N_LAYERS=4 OUT_NORM_GROUPS=0 HIDDEN_NORM_GROUPS=0
((USE_IUFNO_EXPERTS)) && unset K_MODES # ensure K_MODES is max

## load experiment settings
#source experiment_configs/control.sh
#source experiment_configs/recursive_steps.sh
#source experiment_configs/mixture_of_experts.sh
#source experiment_configs/mixture_of_experts_normalized.sh
source experiment_configs/everything.sh # NO-OP b/c everything is already the default
#source experiment_configs/everything_for_WNO_comparison.sh # 800 epochs
#source experiment_configs/WNO.sh # runs for *800* epochs on 64x16x64
#source experiment_configs/WNO+LN.sh # runs for *800* epochs on 64x16x64

# optional modifiers; first one is 'small' jobs that fit on one GPU (incompatible with WNO striding)
[[ "$STRIDE" != '[1.609,1.625,1.203]' ]] && { export BATCH_SIZE=$((BATCH_SIZE*4)) STRIDE=1.47 FAST_DATALOADERS=1; } # simulate 4 nodes, S=1.47 for 68 GB mem target & near 88% utilization
#export HIDDEN_NORM_GROUPS=1
#export OUT_NORM_GROUPS=1
#export SEED=$RANDOM # random behavior (but consistent splitting)
#export MAX_EPOCHS=10

## IMPORTANT: you NEED to use `python -m pytorch_lightning.utilities.consolidate_checkpoint epoch=X-step=Y.ckpt` in order to get a useable model file for resuming or prediction
## Once you do that you can pass the checkpoint path like below to resume a run to get even better performance than what was achievable in the first run
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_RLoP_Extension/477516/checkpoints/epoch=206-step=7452.ckpt.consolidated'

# Run your PyTorch Lightning script
srun --kill-on-bad-exit python -O train.py
#srun --kill-on-bad-exit python -m pdb train.py

# Optional: Debugging info
echo "Job finished at $(date)"
echo done with job time remaining:
squeue -h -j $SLURM_JOB_ID -O TimeLeft
