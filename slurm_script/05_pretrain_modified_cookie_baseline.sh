#!/bin/bash
#SBATCH --job-name=sphinx-pretrain-cookie
#SBATCH --output=/local/s4176650/sphinx/slurm_script/%x-%j.out
#SBATCH --error=/local/s4176650/sphinx/slurm_script/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium

source /local/s4176650/sphinx/slurm_script/_common_sphinx.sh

ENV_NAME="MiniGrid-Modified-Cookie-9x9-v0"
DATA_PATH="collect_${ENV_NAME}.pt"
SEED=1

echo "[1/2] Representation pretraining for ${ENV_NAME}"
python3 pretrain_representations.py \
    --env "${ENV_NAME}" \
    --seed "${SEED}" \
    --data-path "${DATA_PATH}" \
    --epochs 100 \
    --batch-size 500 \
    --beta 0.03 \
    --dynamics-loss-s-coef 0.1 \
    --dynamics-loss-o-coef 0.1 \
    --reward-loss-coef 300.

echo "[2/2] Belief VAE pretraining for ${ENV_NAME}"
python3 -m pretrain_vae \
    --algo belief_vae \
    --env "${ENV_NAME}" \
    --seed "${SEED}" \
    --save-interval 50 \
    --epochs_g 3000 \
    --lr-g 0.0003 \
    --latent-dim-f 16 \
    --latent-dim-vae 64 \
    --data-path "${DATA_PATH}"

echo "Finished at: $(date)"
