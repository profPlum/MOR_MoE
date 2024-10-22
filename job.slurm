#!/bin/bash
#SBATCH --job-name=OneCycle_Clip=5e-3_Trig
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --exclusive
#SBATCH --nodes=8                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS)
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
# (VI) 858 epochs takes 24 hours (with 20 nodes, measured 10/21/24)
export MAX_EPOCHS=444
export LR=$VI_LR # this is 4x higher than theory suggests (it is only possible b/c grad clip)
export GRAD_CLIP=5e-3 # 5e-3 is adjusted based on small hickup at 6.35e-3
export PRIOR_SIGMA=0.2
export OPTIM='Adam' # RAdam is better for resuming
export RLoP=0
export ONE_CYCLE=$((! RLoP)) # we can set both to false to use CosineWarmRestarts, but lets keep it simple
export THREE_PHASE=0
export VI=0 # use VI or not
export TRIG_ENCODINGS=1
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/best_model.ckpt'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/VI_Cos_Extension_Clip=5e-3_LR=1.25e-4/442660/checkpoints/epoch=473-step=18960.ckpt.consolidated'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/RLoP_Extension_Clip=5e-3_LR=standard/418542/checkpoints/epoch=537-step=86080.ckpt.consolidated'

# Run your PyTorch Lightning script
srun --kill-on-bad-exit python -O channel.py

# Optional: Debugging info
echo "Job finished at $(date)"
