#!/bin/bash
#SBATCH --job-name=main-cvae-pretrain
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

MODEL_NAME="${ENV_NAME}_cvae_v6_s${SEED}"
REP_MODEL="storage/${ENV_NAME}_last5_s${SEED}/final_model.pt"

echo "=== CVAE-v6 pretrain: ${ENV_NAME}, seed=${SEED} ==="
echo "Rep model: ${REP_MODEL}"

python3 pretrain_cvae_mikasa_v6.py \
    --env "${ENV_NAME}" \
    --data-path "data/collect_${ENV_NAME}.pt" \
    --model "${MODEL_NAME}" \
    --save-dir storage_cvae \
    --save-interval 50 \
    --seed "${SEED}" \
    --beta ${CVAE_BETA} \
    --state-sentinel-replace 1.0 \
    --lambda-action ${CVAE_LAMBDA_ACTION} \
    --epochs ${CVAE_EPOCHS} \
    --lr ${CVAE_LR} \
    --latent-dim ${CVAE_LATENT_DIM} \
    --batch-size ${CVAE_BATCH_SIZE} \
    --rep-model-path "${REP_MODEL}"

echo "Finished at: $(date)"
