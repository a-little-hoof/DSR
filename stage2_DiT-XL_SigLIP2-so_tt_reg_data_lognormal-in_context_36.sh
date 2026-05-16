#!/bin/bash
# Example SLURM launcher for sampling from the SigLIP2-So400M + DiT-XL test-time-register variant.
# Adjust the SBATCH directives for your cluster, or strip them out and run
# directly under torchrun.
#SBATCH --job-name=stage2_DiT-XL_SigLIP2-so_tt_reg_data_lognormal-in_context_36_twice
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=0-02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out

# conda activate rae_final

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/sample_ddp_multiple_iterations.py \
  --config  configs/stage2/sampling/ImageNet256/DiT-XL_SigLIP2-so_tt_reg_data_lognormal-in_context_36_twice.yaml \
  --sample-dir sample_output/DiT-XL_SigLIP2-so_tt_reg_data_lognormal-in_context_36_twice \
  --precision bf16 \
  --label-sampling equal