#!/bin/bash
#SBATCH --job-name=main-vae-pretrain
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium
#SBATCH --array=0-17
# 6 tasks x 3 seeds = 18 jobs

source /local/s4176650/sphinx/experiments/scripts/main/_common.sh
init_conda believer
cd "${SPHINX_DIR}"

TASK_ID=$((SLURM_ARRAY_TASK_ID / 3))
SEED_ID=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_ID]}

parse_env_config "${ALL_ENVS[$TASK_ID]}"

MODEL_NAME="${ENV_NAME}_vae_s${SEED}"
REP_MODEL="storage/${ENV_NAME}_last5_s${SEED}/final_model.pt"

echo "=== VAE (Believer) pretrain: ${ENV_NAME}, seed=${SEED} ==="
echo "Rep model: ${REP_MODEL}"

python3 pretrain_vae_mikasa.py \
    --env "${ENV_NAME}" \
    --data-path "data/collect_${ENV_NAME}.pt" \
    --model "${MODEL_NAME}" \
    --save-dir storage \
    --save-interval 50 \
    --seed "${SEED}" \
    --beta ${VAE_BETA} \
    --state-sentinel-replace 1.0 \
    --epochs ${VAE_EPOCHS} \
    --lr ${VAE_LR} \
    --latent-dim ${VAE_LATENT_DIM} \
    --batch-size ${VAE_BATCH_SIZE} \
    --rep-model-path "${REP_MODEL}"

echo "Finished at: $(date)"
