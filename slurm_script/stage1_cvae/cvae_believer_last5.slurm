#!/bin/bash
#SBATCH --job-name=sphinx-pretrain-rep
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%j.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=49:00:00
# #SBATCH --nodelist=saronite
#SBATCH --nodelist=netherite
# #SBATCH --nodelist=ceratanium
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



python3 pretrain_cvae_mikasa.py \
    --env RememberShapeAndColor3x2-v0 \
    --data-path data/collect_RememberShapeAndColor3x2-v0.pt \
    --model RememberShapeAndColor3x2-v0_cvae_mikasa_last5_ep3000 \
    --save-dir storage_cvae \
    --save-interval 50 \
    --beta 1 \
    --state-sentinel-replace 1.0 \
    --lambda-action 1.0 \
    --epochs 3000 \
    --lr 0.0003 \
    --latent-dim 32 \
    --batch-size 256 \
    --rep-model-path storage/RememberShapeAndColor3x2-v0_last5/final_model.pt
