#!/bin/bash
#SBATCH --job-name=VI_RLoP_Extension_Clip=5e-3_LR=1.25e-4
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --exclusive
#SBATCH --nodes=16                           # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --time=3-00:00:00                    # Maximum runtime (HH:MM:SS)
#SBATCH --partition=glinda                   # Partition name
#SBATCH --exclude gn78,gn71 #gn111,gn16 #both these nodes have caused timeouts before, maybe exclude?

# Not going to work anymore b/c we would need to save the entire checkpoint which causes OOM...
# ##SBATCH --signal=SIGUSR1@90

# Load modules or environment
. ~/.bashrc
conda activate uqops+proxy
module load cuda/11.8.0

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun nvidia-smi

# suggested by Adam default lr=0.001/32 s.t. 32 is default batch size
STANDARD_LR=3.125e-5

# (VI) 180 epochs takes 24 hours (with 8 nodes, measured 10/2/24)
export MAX_EPOCHS=540
export LR=1.25e-4 # this is 4x higher than theory suggests (it is only possible b/c grad clip)
export GRAD_CLIP=5e-3 # 5e-3 is adjusted based on small hickup at 6.35e-3
export PRIOR_SIGMA=0.2
export OPTIM='Adam'
export RLoP=1
export ONE_CYCLE=$((! RLoP)) # we can set both to false to use CosineWarmRestarts, but lets keep it simple
export THREE_PHASE=0
export VI=1 # use VI or not
export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/VI_RLoP_Extension_Clip=1_LR=standard/416831/checkpoints/epoch=82-step=6640.ckpt.consolidated'
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/OneCycle_Clip=2e-3/414092/checkpoints/epoch=749-step=60000.ckpt'

# Run your PyTorch Lightning script
srun python -O channel.py

# Optional: Debugging info
echo "Job finished at $(date)"
