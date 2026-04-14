#!/bin/bash
# Common configuration for all main experiment scripts
# Source this file at the top of each SLURM script.

set -eo pipefail
export PYTHONUNBUFFERED=1

# ---- Paths ----
SPHINX_DIR="/local/s4176650/sphinx"
MIKASA_DIR="/local/s4176650/MIKASA-Robo"

# ---- Seeds ----
SEEDS=(33 42 99)

# ---- Task definitions ----
#   ENV_NAME | NUM_STEPS | NUM_EVAL_STEPS | EVAL_FREQ
# Remember-type (short episodes)
REMEMBER_ENVS=(
    "RememberColor9-v0|60|720|48"
    "RememberShapeAndColor3x2-v0|60|720|48"
    "RememberShape9-v0|60|720|48"
    "RememberShapeAndColor3x3-v0|60|720|48"
)
# Intercept-type (long episodes)
INTERCEPT_ENVS=(
    "InterceptFast-v0|90|1080|25"
    "InterceptMedium-v0|90|1080|25"
)
ALL_ENVS=("${REMEMBER_ENVS[@]}" "${INTERCEPT_ENVS[@]}")

# ---- Unified PPO parameters ----
GAMMA=0.99
GAE_LAMBDA=0.9
ENT_COEF=0.001
TARGET_KL=0.05
NUM_ENVS=256
NUM_EVAL_ENVS=16
LR=1e-4
UPDATE_EPOCHS=2
TOTAL_TIMESTEPS=10_000_000
NUM_MINIBATCHES=32

# ---- Stage1 parameters ----
# Representation pretraining
REP_EPOCHS=200
REP_BATCH_SIZE=500
REP_BETA=0.03
REP_LATENT_DIM=16

# CVAE-v6 pretraining
CVAE_EPOCHS=6000
CVAE_LR=0.0003
CVAE_LATENT_DIM=32
CVAE_BATCH_SIZE=256
CVAE_BETA=1
CVAE_LAMBDA_ACTION=1.0

# VAE pretraining
VAE_EPOCHS=6000
VAE_LR=0.0003
VAE_LATENT_DIM=32
VAE_BATCH_SIZE=256
VAE_BETA=1

# Data collection
DATA_NUM_EPISODES=5000
DATA_BATCH_SIZE=128

# ---- LSTM/GRU parameters ----
RNN_HIDDEN_SIZE=512
RNN_NUM_LAYERS=1
RNN_DROPOUT=0.0

# ---- Helper functions ----
parse_env_config() {
    # Usage: parse_env_config "RememberColor9-v0|60|720|48"
    IFS='|' read -r ENV_NAME NUM_STEPS NUM_EVAL_STEPS EVAL_FREQ <<< "$1"
}

init_conda() {
    local env_name="${1:-believer}"
    # Temporarily disable exit-on-error: source ~/.bashrc may trigger
    # non-fatal errors (e.g., D-Bus "Failed to connect to bus" on compute nodes)
    set +e
    source ~/.bashrc 2>/dev/null
    set -e
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    hostname
    nvidia-smi -L || true
}
