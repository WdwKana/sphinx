#!/bin/bash

#SBATCH --job-name=mikasa-rc-belief
#SBATCH --output=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%j.out
#SBATCH --error=/local/s4176650/MIKASA-Robo/slurm_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --nodelist=netherite

mkdir -p /local/s4176650/MIKASA-Robo/slurm_logs

source ~/.bashrc
conda activate mikasa

cd /local/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1

# ---- RememberShapeAndColor3x2 with Belief ----
for seed in 33 42 99; do
  echo "Running RememberShapeAndColor3x2-v0 with belief, seed ${seed}"
  python3 baselines/ppo/ppo_memtasks_back2.py \
    --env_id=RotateStrictPosNeg-v0 \
    --exp-name=ppo-last5-v2-baseline-${seed} \
    --save-model \
    --num-steps=90 \
    --num_eval_steps=1080 \
    --include-rgb \
    --include-joints \
    --include-belief \
    --belief-path=/local/s4176650/sphinx/storage/RotateStrictPosNeg-v0_vae_mikasa_last5_v2/final_vae_model.pt \
    --seed=$seed \
    --total-timesteps=20_000_000 \
    --eval-freq=25 \
    --num-envs=256 \
    --num-minibatches=8 
#--eval_belief_ablation
done
