#!/bin/bash
#SBATCH --job-name=main-rep-pretrain
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium
#SBATCH --array=0-17
# 6 tasks x 3 seeds = 18 jobs
# index = task_id * 3 + seed_id

source /local/s4176650/sphinx/experiments/scripts/main/_common.sh
init_conda believer
cd "${SPHINX_DIR}"

TASK_ID=$((SLURM_ARRAY_TASK_ID / 3))
SEED_ID=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_ID]}

parse_env_config "${ALL_ENVS[$TASK_ID]}"

MODEL_NAME="${ENV_NAME}_last5_s${SEED}"

echo "=== Rep pretrain: ${ENV_NAME}, seed=${SEED} ==="

python3 pretrain_representations_mikasa.py \
    --env "${ENV_NAME}" \
    --model "${MODEL_NAME}" \
    --data-path "data/collect_${ENV_NAME}.pt" \
    --seed "${SEED}" \
    --epochs ${REP_EPOCHS} \
    --state-sentinel-replace 1.0 \
    --batch-size ${REP_BATCH_SIZE} \
    --latent-dim ${REP_LATENT_DIM} \
    --beta ${REP_BETA} \
    --dynamics-loss-s-coef 0.3 \
    --dynamics-loss-o-coef 0.03 \
    --reward-loss-coef 10

echo "Finished at: $(date)"
