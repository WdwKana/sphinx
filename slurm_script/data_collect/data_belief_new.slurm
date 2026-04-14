#!/bin/bash
#SBATCH --job-name=collect-random-data
#SBATCH --output=/local/s4176650/sphinx/slurm_logs/%x-%j.out
#SBATCH --error=/local/s4176650/sphinx/slurm_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=49:00:00
# #SBATCH --nodelist=netherite
# #SBATCH --nodelist=saronite
#SBATCH --nodelist=ceratanium
# 可按需加: #SBATCH --partition=gpu

set -eo pipefail

export PYTHONUNBUFFERED=1

# 项目路径
SPHINX_ROOT=/local/s4176650/sphinx

# 注意：这两行对 SBATCH 的 output/error 没用（output/error 在脚本执行前就要目录存在）
# 但对你脚本里其它写文件操作有用
mkdir -p "${SPHINX_ROOT}/slurm_logs"
mkdir -p "${SPHINX_ROOT}/data"

# conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate believer

hostname
nvidia-smi -L || true

# 进入 sphinx 项目目录
cd "${SPHINX_ROOT}"

python -c "import mikasa_robo_suite; print(mikasa_robo_suite.__file__)"
# 运行随机策略数据收集（数据会保存到 /local/s4176650/sphinx/data/sphinx_format/ 下）
python "${SPHINX_ROOT}/get_random_datasets_full_state.py" \
    --env-id InterceptGrabFast-v0 \
    --path-to-save-data "${SPHINX_ROOT}/data" \
    --num-episodes 5000 \
    --batch-size 128 \
    --seed 42
