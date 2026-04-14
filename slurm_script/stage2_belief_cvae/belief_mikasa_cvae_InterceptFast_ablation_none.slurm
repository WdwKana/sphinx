#!/bin/bash

#SBATCH --job-name=mikasa-interceptmedium-belief
#SBATCH --output=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%j.out
#SBATCH --error=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --nodelist=netherite
# #SBATCH --nodelist=saronite

mkdir -p /local/s4176650/MIKASA-Robo/slurm_logs

source ~/.bashrc
conda activate mikasa

cd /local/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1

BELIEF_CKPT="/local/s4176650/sphinx/storage_cvae/InterceptFast-v0_cvae_mikasa_last5_v6_001/final_vae_model.pt"

for seed in 33 42 99; do
  echo "Running Intercept-Medium (BELIEF), seed ${seed}"
  python3 baselines/ppo/ppo_memtasks_cvae.py \
    --env-id=InterceptFast-v0 \
    --exp-name=ppo-cvae-v6-001-interceptfast-${seed} \
    --save-model \
    --num-steps=90 \
    --num-eval-steps=1080 \
    --include-rgb \
    --include-joints \
    --include-belief \
    --belief-path="${BELIEF_CKPT}" \
    --seed="${seed}" \
    --total-timesteps=10_000_000 \
    --eval-freq=25 \
    --no-finite-horizon-gae \
    --gae-lambda=0.9 \
    --gamma=0.99 \
    --ent_coef=0.001 \
    --target-kl=0.05 \
    --num-eval-envs=16 \
    --num-envs=256 \
    --learning-rate=1e-4 \
    --update-epochs=2
done
