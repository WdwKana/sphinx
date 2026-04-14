#!/bin/bash
#SBATCH --job-name=sphinx-collect-all
#SBATCH --output=/local/s4176650/sphinx/slurm_script/%x-%j.out
#SBATCH --error=/local/s4176650/sphinx/slurm_script/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=49:00:00
#SBATCH --nodelist=ceratanium

source /local/s4176650/sphinx/slurm_script/_common_sphinx.sh

SEED=1

echo "[1/5] Collecting MiniGrid-Genie-8x8-v0"
python3 collect.py --env MiniGrid-Genie-8x8-v0 --episodes 3000 --seed "${SEED}"

echo "[2/5] Collecting MiniGrid-NoisyTV-Genie-8x8-v0"
python3 collect.py --env MiniGrid-NoisyTV-Genie-8x8-v0 --episodes 3000 --seed "${SEED}"

echo "[3/5] Collecting MiniGrid-Lying-Genie-8x8-v0"
python3 collect.py --env MiniGrid-Lying-Genie-8x8-v0 --episodes 3000 --seed "${SEED}"

echo "[4/5] Collecting MiniGrid-Modified-Cookie-9x9-v0"
python3 collect_modified_cookie.py --env MiniGrid-Modified-Cookie-9x9-v0 --episodes 1000 --seed "${SEED}"

echo "[5/5] Collecting EscapeRoom-v0"
python3 collect_escape.py --env EscapeRoom-v0 --episodes 500 --seed "${SEED}"

echo "Finished at: $(date)"
