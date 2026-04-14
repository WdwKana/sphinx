#!/bin/bash

set -eo pipefail

export PYTHONUNBUFFERED=1

# Some cluster bash init scripts read unset variables directly, so delay
# nounset until after shell/conda initialization completes.
set +u
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate believer
set -u

PROJECT_DIR="${SPHINX_PROJECT_DIR:-/local/s4176650/sphinx}"
if [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="/home/s4176650/sphinx"
fi

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
nvidia-smi -L || true

cd "$PROJECT_DIR"
