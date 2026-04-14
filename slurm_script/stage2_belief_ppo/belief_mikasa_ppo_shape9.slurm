#!/bin/bash

#SBATCH --job-name=mikasa-rc-belief
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

# ---- RememberColor with Belief ----
for seed in 33 42; do
  echo "Running RememberShape9-v0 with belief, seed ${seed}"
  python3 baselines/ppo/ppo_memtasks_back2.py \
    --env_id=RememberShape9-v0 \
    --exp-name=ppo-baseline-last5-${seed} \
    --capture-video \
    --save-model \
    --num-steps=60 \
    --num_eval_steps=720 \
    --include-rgb \
    --include-joints \
    --include-belief \
    --belief-path=/local/s4176650/sphinx/storage/RememberShape9-v0_vae_mikasa_last5/final_vae_model.pt \
    --seed=$seed \
    --total-timesteps=10_380_000 \
    --no-finite-horizon-gae \
    --eval-freq=48 \
    --gae-lambda=0.9 \
    --gamma=0.8 \
    --num-eval-envs=16 \
    --num-envs=200 \
    --update-epochs=4
#--eval_belief_ablation
done
