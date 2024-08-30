#!/bin/bash
#SBATCH --job-name=channel_MOR_MoE           # Job name
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --exclusive
#SBATCH --nodes=20                           # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --time=6:00:00                       # Maximum runtime (HH:MM:SS)
#SBATCH --partition=glinda                   # Partition name
#SBATCH --exclude gn78

# Load modules or environment
. ~/.bashrc
conda activate uqops+proxy
module load cuda/11.8.0
export NCCL_DEBUG=INFO

# Optional: Debugging info
echo "Job started on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "batch node ip: $(host $(hostname))"
echo "rank hostnames & Ips:"
srun bash -c 'echo host: $SLURMD_NODENAME, rank: $PMI_RANK, slurm_proc_id: $SLURM_PROCID, $(host $(hostname))'
echo srun nvidia-smi:
srun nvidia-smi

# Run your PyTorch Lightning script
srun python channel.py

# Optional: Debugging info
echo "Job finished at $(date)"

