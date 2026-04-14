#!/bin/bash
#SBATCH --job-name=sphinx-pretrain-rep
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%j.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=49:00:00
# #SBATCH --nodelist=saronite
# #SBATCH --nodelist=netherite
#SBATCH --nodelist=ceratanium
# 可按需加: #SBATCH --partition=gpu

set -eo pipefail

# 目录
mkdir -p /local/s4176650/ReinFlow/slurm_logs
# mkdir -p /local/s4176650/ReinFlow/wandb_offline # Sphinx不一定用wandb，按需保留
# mkdir -p /local/s4176650/ReinFlow/logs
# mkdir -p /local/s4176650/ReinFlow/data

# 环境变量
export PYTHONUNBUFFERED=1

# conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate believer

hostname
nvidia-smi -L || true

# 切换到 sphinx 目录
cd /local/s4176650/sphinx

# 运行 Sphinx Representation Learning Pretraining
python pretrain_representations_mikasa.py \
    --env RotateStrictPos-v0 \
    --model RotateStrictPos-v0_last5 \
    --data-path /local/s4176650/sphinx/data/collect_RotateStrictPos-v0.pt \
    --epochs 200 \
    --state-sentinel-replace 1.0 \
    --batch-size 500 \
    --latent-dim 16 \
    --beta 0.03 \
    --dynamics-loss-s-coef 0.3 \
    --dynamics-loss-o-coef 0.03 \
    --reward-loss-coef 10
