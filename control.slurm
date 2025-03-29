#!/bin/bash
#SBATCH --job-name=control                   # Job name
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --exclusive
#SBATCH --nodes=18                           # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --time=1-03:00:00                     # Maximum runtime (HH:MM:SS)
#SBATCH --partition=glinda                   # Partition name
#SBATCH --exclude gn78,gn71,gn54,gn60 # gn54 failed in srun echo hi... gn60 failed with unknown error, idk if others are still problematic 

# Not going to work anymore b/c we would need to save the entire checkpoint which causes OOM...
# ##SBATCH --signal=SIGUSR1@90

# Load modules or environment
. ~/.bashrc
conda activate uqops+proxy
module load cuda/11.8.0

#export NCCL_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=INFO

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun nvidia-smi

STANDARD_LR=3.125e-5 # theoretically suggested by Adam default lr=0.001/32 s.t. 32 is default batch size
VI_LR=1.25e-4 # this is 4x higher than theory suggests (it is only possible b/c grad clip)
MLE_LR=2.5e-4 # this is 8x higher than theory suggests (it is only possible b/c grad clip + determinism)

# (MLE) 582 epochs takes 24 hours (with 8 nodes, measured 10/22/24)
# (VI) 515 epochs takes 24 hours (with 12 nodes, measured 10/22/24)
# (VI) 644 epochs takes 24 hours (with 20 nodes, measured 10/23/24)
# (VI) 445 epochs takes 24 hours (with 10 nodes, measured 10/26/24)
# (VI+TRIG) 333 epochs takes 24 hours (with 10 nodes, measured 10/26/24)
# (VI+TRIG) 678 epochs takes 24 hours (with 20 nodes, measured 10/31/24)
export MAX_EPOCHS=333 # MUST BE ACCURATELY ADJUSTED based on estimated epochs (this ensures that OneCycle is properly calibrated)
export LR=$VI_LR # this is 4x higher than theory suggests (it is only possible b/c grad clip)
export GRAD_CLIP=2.5e-3 # it is best to keep this 2.5e-3 value, it will automatically be adjusted based on number of recursive timesteps
export PRIOR_SIGMA=0.2 # approximately reflects the values of sigma at initialization
export OPTIM='Adam' # RAdam is better for resuming, otherwise use Adam
export RLoP=0 # Reduce Learning-rate on Plateu, can be useful for resuming (but otherwise keep it off!)
export ONE_CYCLE=$((! RLoP)) # OneCycle warmup & decay lr schedule. Really important for scaling to multiple GPUs!
export THREE_PHASE=0 # faster decay in one cycle, usually not needed
export VI=1 # use VI or not (in bash binary booleans have the best support for boolean logic)
export TRIG_ENCODINGS=1 # trig encodings slow down training almost 1/3 but might give better partitions?
export TOTAL_VARIANCE=0 # whether to use total variance for NLL
export TIME_CHUNKING=2 # 9 is max (but only verified to work for 10 at least nodes)
export N_LAYERS=8 # number of expert layers
export N_EXPERTS=1 # number of experts

## IMPORTANT: you NEED to use `python -m pytorch_lightning.utilities.consolidate_checkpoint epoch=X-step=Y.ckpt` in order to get a useable model file for resuming or prediction
## Once you do that you can pass the checkpoint path like below to resume a run to get even better performance than what was achievable in the first run
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_RLoP_Extension/477516/checkpoints/epoch=206-step=7452.ckpt.consolidated'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_RLoP_Extension/477516/checkpoints/epoch=114-step=4140.ckpt.consolidated'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_OneCycle_RecursiveClip=2.5e-3/467132/checkpoints/epoch=867-step=31248.ckpt.consolidated'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/VI_OneCycle_RecursiveClip=2.5e-3_Trig/465271/checkpoints/epoch=242-step=3888.ckpt.consolidated'

# Run your PyTorch Lightning script
srun --kill-on-bad-exit python -O channel.py

# Optional: Debugging info
echo "Job finished at $(date)"
