#!/bin/bash
#SBATCH --job-name=main-s2-gru
#SBATCH --output=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%A_%a.out
#SBATCH --error=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --nodelist=ceratanium
#SBATCH --array=0-17
# 6 tasks x 3 seeds = 18 jobs

source /local/s4176650/sphinx/experiments/scripts/main/_common.sh
init_conda mikasa
cd "${MIKASA_DIR}"
mkdir -p "${MIKASA_DIR}/slurm_logs"

TASK_ID=$((SLURM_ARRAY_TASK_ID / 3))
SEED_ID=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_ID]}

parse_env_config "${ALL_ENVS[$TASK_ID]}"

echo "=== Stage2 PPO+GRU: ${ENV_NAME}, seed=${SEED} ==="

python3 baselines/ppo/ppo_memtasks_gru.py \
    --env-id="${ENV_NAME}" \
    --exp-name="ppo-gru-${ENV_NAME}-${SEED}" \
    --save-model \
    --num-steps="${NUM_STEPS}" \
    --num-eval-steps="${NUM_EVAL_STEPS}" \
    --include-rgb \
    --include-joints \
    --seed="${SEED}" \
    --total-timesteps=${TOTAL_TIMESTEPS} \
    --no-finite-horizon-gae \
    --eval-freq="${EVAL_FREQ}" \
    --gae-lambda=${GAE_LAMBDA} \
    --gamma=${GAMMA} \
    --ent-coef=${ENT_COEF} \
    --target-kl=${TARGET_KL} \
    --num-eval-envs=${NUM_EVAL_ENVS} \
    --num-envs=${NUM_ENVS} \
    --learning-rate=${LR} \
    --update-epochs=${UPDATE_EPOCHS} \
    --num-minibatches=${NUM_MINIBATCHES} \
    --gru-hidden-size=${RNN_HIDDEN_SIZE} \
    --gru-num-layers=${RNN_NUM_LAYERS} \
    --gru-dropout=${RNN_DROPOUT}

echo "Finished at: $(date)"
