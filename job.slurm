#!/bin/bash
#SBATCH --output=logs/%x_%j.out              # Output file (%x: job-name, %j: job-id)
#SBATCH --error=logs/%x_%j.err               # Error file (%x: job-name, %j: job-id)
#SBATCH --nodes=1                            # Trying allocating a whole node to play nice with pytorch-lightning
#SBATCH --ntasks-per-node=4                  # Tasks per node (usually 1 for PyTorch)
#SBATCH --mem-per-gpu=120000                 # CPU? RAM per GPU < 1/4 of the total node RAM = 120GB
#SBATCH --gpus-per-node=4                    # GPUs per task is 1
#SBATCH --cpus-per-task=72                   # CORES per task for data loaders (72 avail per GPU)
#SBATCH --time=1-00:00:00                    # Maximum runtime (HH:MM:SS)
#SBATCH --partition=ghx4                     # Partition name
#SBATCH --account=beoi-dtai-gh
#SBATCH --job-name=CNN-K6-scaled-LL

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

## Kahuna measurements:
# (MLE) 582 epochs takes 24 hours (with 8 nodes, measured 10/22/24)
# (VI) 515 epochs takes 24 hours (with 12 nodes, measured 10/22/24)
# (VI) 644 epochs takes 24 hours (with 20 nodes, measured 10/23/24)
# (VI) 445 epochs takes 24 hours (with 10 nodes, measured 10/26/24)
# (VI+TRIG) 333 epochs takes 24 hours (with 10 nodes, measured 10/26/24)
# (VI+TRIG) 678 epochs takes 24 hours (with 20 nodes, measured 10/31/24)

# (VI+TRIG) 1550 epochs takes 20 hours (with 4 GH200s, measured 4/2/25)
export MAX_EPOCHS=1600 #3100 # MUST BE ACCURATELY ADJUSTED based on estimated epochs (this ensures that OneCycle is properly calibrated)
export LR=$VI_LR # this is 4x higher than theory suggests (it is only possible b/c grad clip)
export GRAD_CLIP=8.944e-2 # it is best to keep this 8.944e-2 value, it will automatically be adjusted based on number of recursive timesteps & batch_size
export PRIOR_SIGMA=0.2 # approximately reflects the values of sigma at initialization
export OPTIM='Adam' # RAdam is better for resuming, otherwise use Adam
export RLoP=0 # Reduce Learning-rate on Plateu, can be useful for resuming (but otherwise keep it off!)
export ONE_CYCLE=$((! RLoP)) # OneCycle warmup & decay lr schedule. Really important for scaling to multiple GPUs!
export THREE_PHASE=0 # faster decay in one cycle, usually not needed
export VI=1 # use VI or not (in bash binary booleans have the best support for boolean logic)
export TRIG_ENCODINGS=1 # gives better (nearly periodic) partitions
export TIME_CHUNKING=10 # 10 is default now b/c we can fit it in GH200s
export BATCH_SIZE=3 # GH200 can handle batch_size=3! (it has > x2 the memory of A100)
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden filters (aka channel size)
export N_EXPERTS=3 # number of experts

# CNN experts?
export USE_CNN_EXPERTS=1
export CNN_FILTER_SIZE=6
#export CNN_FILTER_SIZE='[103,26,77]' # match k_modes...
export N_LAYERS=12 N_EXPERTS=1 #LR=1e-7

## load experiment settings
#source experiment_configs/control.sh
#source experiment_configs/recursive_steps.sh
#source experiment_configs/mixture_of_experts.sh
#source experiment_configs/everything.sh # NO-OP b/c everything is already the default

#export SEED=$RANDOM # random behavior (but consistent splitting)
#export BATCH_SIZE=27 # use same number of outputs
#export N_EXPERTS=1
#export MAX_EPOCHS=5

## IMPORTANT: you NEED to use `python -m pytorch_lightning.utilities.consolidate_checkpoint epoch=X-step=Y.ckpt` in order to get a useable model file for resuming or prediction
## Once you do that you can pass the checkpoint path like below to resume a run to get even better performance than what was achievable in the first run
#export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_RLoP_Extension/477516/checkpoints/epoch=206-step=7452.ckpt.consolidated'

# Run your PyTorch Lightning script
srun --kill-on-bad-exit python -O channel.py
#srun --kill-on-bad-exit python -m pdb channel.py

# Optional: Debugging info
echo "Job finished at $(date)"
echo done with job time remaining:
squeue -h -j $SLURM_JOB_ID -O TimeLeft
