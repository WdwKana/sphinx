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



python3 pretrain_vae_mikasa.py \
    --env InterceptGrabMedium-v0 \
    --data-path data/collect_InterceptGrabMedium-v0.pt \
    --model InterceptGrabMedium-v0_vae_mikasa_last5 \
    --save-dir storage \
    --state-sentinel-replace 1.0 \
    --save-interval 50 \
    --beta 1 \
    --epochs 10000 \
    --lr 0.0003 \
    --latent-dim 32 \
    --batch-size 256 \
    --rep-model-path storage/InterceptGrabMedium-v0_last5/final_model.pt

#python3 -m pretrain_vae --algo belief_vae --env EscapeRoom-v0 --model EscapeRoom-v0_2 --save-interval 50 --batch-size-g 100 --epochs_g 5000 --lr-g 0.0003 --latent-dim-f 16 --data-path collect_EscapeRoom-v0.pt

#python3 -m train --algo belief_vae --env EscapeRoom-v0 --save-interval 50 --frames 5000000 --procs 32 --recurrence 1 --frames_per_proc 256 --batch-size 2048 --epochs 8 --batch-size-g 2048 --epochs_g 8 --lr-g 0.0003 --lr 0.0005 --entropy-coef 0.01 --latent-dim-vae 32 --latent-dim-f 16
