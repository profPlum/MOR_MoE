#!/bin/bash
#SBATCH --job-name=OneCycle3_Clip=2e-3
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --exclusive
#SBATCH --nodes=8                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS)
#SBATCH --partition=glinda                   # Partition name
#SBATCH --exclude gn78,gn71 #gn111,gn16 #both these nodes have caused timeouts before, maybe exclude?

# Load modules or environment
. ~/.bashrc
conda activate uqops+proxy
module load cuda/11.8.0
#export NCCL_DEBUG=INFO

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun nvidia-smi

# 500 epochs takes about 21.5 hours (with 8 nodes, measured: 9/12/24)
# 100 epochs takes about 9 hours (with 4 nodes, measured 9/16/24)
export MAX_EPOCHS=500
export LR=2.5e-4
export GRAD_CLIP=6.35e-3
export PRIOR_SIGMA=0.2
export OPTIM='Adam'
export RLoP=0
export ONE_CYCLE=$((! RLoP)) # we can set both to false to use CosineWarmRestarts, but lets keep it simple
export THREE_PHASE=1
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/OneCycle_Adam+Clip=5_POU/version_0/checkpoints/epoch=239-step=19200.ckpt'

# Run your PyTorch Lightning script
srun python channel.py

# Optional: Debugging info
echo "Job finished at $(date)"
