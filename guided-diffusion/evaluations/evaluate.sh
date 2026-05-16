#!/bin/bash
# Compute FID / IS / Precision / Recall against the ADM reference statistics
# for every epoch directory produced by sample_ddp_multiple_iterations.py.
#SBATCH --job-name=compute_fid
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=0-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=logs/%x-%j.out

# Activate the adm-fid environment described in the top-level README
# (created separately from rae_final because TensorFlow 2.19 and PyTorch
# 2.8 pin incompatible CUDA versions).
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate adm-fid

path_list=(
# ../../sample_output/SigLip2-B_data_lognormal
# ../../sample_output/SigLip2-so_data_lognormal
../../sample_output/DiT-XL_SigLIP2-B_tt_reg_data_lognormal-in_context_36
# ../../sample_output/DiT-XL_SigLIP2-so_tt_reg_data_lognormal-in_context_36_twice
)
for path in "${path_list[@]}"; do
echo "currently we are measuring ${path}"
python evaluator_iteratively.py \
  "imagenet-val-package-correct-no-transform.npz" \
  "${path}"
done


