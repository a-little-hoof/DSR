#!/bin/bash
# Example SLURM launcher for the SigLIP2-B + DiT-XL baseline (800 epochs).
# Adjust the SBATCH directives (account, partition, mail) for your cluster, or
# strip them out and run directly under torchrun.
#SBATCH --job-name=train-stage2_DiT-XL_SigLIP2-B_data_lognormal
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:8
#SBATCH --mem=200G

#SBATCH --output=logs/%x-%j.out

# --- (A) Distributed env ---
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0

# Pick up rendezvous info when running under SLURM
if [ -n "$SLURM_JOB_NODELIST" ]; then
  HOSTLIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  export MASTER_ADDR=$(echo $HOSTLIST | awk '{print $1}')
  export MASTER_PORT=${MASTER_PORT:-29500}
  echo "SLURM_NNODES = $SLURM_NNODES"
  echo "HOSTLIST     = $HOSTLIST"
  echo "MASTER_ADDR  = $MASTER_ADDR"
  echo "MASTER_PORT  = $MASTER_PORT"
fi

# --- (B) Weights & Biases (optional) ---
# Set WANDB_KEY, ENTITY, and PROJECT in your shell before submitting if you
# pass --wandb below. The script will exit cleanly if --wandb is omitted.
export PROJECT="${PROJECT:-RAE-REG-stage2_DiT-XL_SigLIP2-B_data_lognormal}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-train-stage2_DiT-XL_SigLIP2-B_data_lognormal}"

# Activate your environment (assumes you followed the README setup)
# conda activate rae_final

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  src/train.py \
  --config configs/stage2/training/ImageNet256/DiT-XL_SigLIP2-B_data_lognormal.yaml \
  --data-path "$DATA_PATH" \
  --results-dir results/stage2/DiT-XL_SigLIP2-B_data_lognormal \
  --precision fp32 \
  --wandb \
  --compile