#!/bin/bash
#SBATCH --job-name=main-collect-data
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium
#SBATCH --array=0-5
# 6 tasks: 0=Color9, 1=3x2, 2=Shape9, 3=3x3, 4=InterceptFast, 5=InterceptMedium

source /local/s4176650/sphinx/experiments/scripts/main/_common.sh
init_conda believer
cd "${SPHINX_DIR}"
mkdir -p "${SPHINX_DIR}/data"

parse_env_config "${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Collecting data for ${ENV_NAME} ==="

# Data collection uses a fixed seed (42) since it's a random policy.
# The per-seed differentiation starts at representation pretraining.
python3 get_random_datasets_full_state.py \
    --env-id "${ENV_NAME}" \
    --path-to-save-data "${SPHINX_DIR}/data" \
    --num-episodes ${DATA_NUM_EPISODES} \
    --batch-size ${DATA_BATCH_SIZE} \
    --seed 42

echo "Finished at: $(date)"
