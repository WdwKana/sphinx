#!/bin/bash
#SBATCH --job-name=sphinx-train-escape
#SBATCH --output=/local/s4176650/sphinx/slurm_script/%x-%A_%a.out
#SBATCH --error=/local/s4176650/sphinx/slurm_script/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium
#SBATCH --array=0-2

source /local/s4176650/sphinx/slurm_script/_common_sphinx.sh

ENV_NAME="EscapeRoom-v0"
SEEDS=(33 42 99)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Training ${ENV_NAME} with seed ${SEED}"
python3 -m train \
    --algo belief_vae \
    --env "${ENV_NAME}" \
    --seed "${SEED}" \
    --save-interval 50 \
    --frames 5000000 \
    --procs 32 \
    --recurrence 1 \
    --frames_per_proc 256 \
    --batch-size 2048 \
    --epochs 8 \
    --batch-size-g 2048 \
    --epochs_g 8 \
    --lr-g 0.0003 \
    --lr 0.0005 \
    --entropy-coef 0.01 \
    --latent-dim-vae 32 \
    --latent-dim-f 16

echo "Finished at: $(date)"
