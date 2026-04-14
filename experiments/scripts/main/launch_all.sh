#!/bin/bash
# Launch all main experiments in dependency order.
# Usage: bash launch_all.sh [stage]
#   stage=0  Data collection only
#   stage=1  Stage1 pretraining (after data is ready)
#   stage=2  Stage2 PPO training (after Stage1 is ready)
#   stage=all Launch everything with SLURM dependencies (automated)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /local/s4176650/sphinx/slurm_logs
mkdir -p /local/s4176650/MIKASA-Robo/slurm_logs

STAGE="${1:-help}"

case "$STAGE" in
    0)
        echo "=== Stage 0: Data Collection (6 tasks) ==="
        sbatch "${SCRIPT_DIR}/00_collect_data.slurm"
        ;;
    1)
        echo "=== Stage 1: Pretraining ==="
        echo "[1/3] Representation pretraining (18 jobs)"
        JOB_REP=$(sbatch --parsable "${SCRIPT_DIR}/01_rep_pretrain.slurm")
        echo "  Submitted: ${JOB_REP}"

        echo "[2/3] CVAE-v6 pretraining (18 jobs, depends on rep)"
        JOB_CVAE=$(sbatch --parsable --dependency=afterok:${JOB_REP} "${SCRIPT_DIR}/02_cvae_pretrain.slurm")
        echo "  Submitted: ${JOB_CVAE} (after ${JOB_REP})"

        echo "[3/3] VAE pretraining (18 jobs, depends on rep)"
        JOB_VAE=$(sbatch --parsable --dependency=afterok:${JOB_REP} "${SCRIPT_DIR}/03_vae_pretrain.slurm")
        echo "  Submitted: ${JOB_VAE} (after ${JOB_REP})"
        ;;
    2)
        echo "=== Stage 2: PPO Training ==="
        echo "[1/5] CVAE-v6 (Ours) - 18 jobs"
        sbatch "${SCRIPT_DIR}/04_stage2_cvae.slurm"

        echo "[2/5] Believer VAE - 18 jobs"
        sbatch "${SCRIPT_DIR}/04_stage2_vae.slurm"

        echo "[3/5] PPO baseline - 18 jobs"
        sbatch "${SCRIPT_DIR}/04_stage2_ppo.slurm"

        echo "[4/5] PPO+LSTM - 18 jobs"
        sbatch "${SCRIPT_DIR}/04_stage2_lstm.slurm"

        echo "[5/5] PPO+GRU - 18 jobs"
        sbatch "${SCRIPT_DIR}/04_stage2_gru.slurm"
        ;;
    all)
        echo "=== Full Pipeline with Dependencies ==="
        echo "[0] Data collection"
        JOB_DATA=$(sbatch --parsable "${SCRIPT_DIR}/00_collect_data.slurm")
        echo "  Submitted: ${JOB_DATA}"

        echo "[1] Rep pretrain (after data)"
        JOB_REP=$(sbatch --parsable --dependency=afterok:${JOB_DATA} "${SCRIPT_DIR}/01_rep_pretrain.slurm")
        echo "  Submitted: ${JOB_REP}"

        echo "[2] CVAE-v6 pretrain (after rep)"
        JOB_CVAE=$(sbatch --parsable --dependency=afterok:${JOB_REP} "${SCRIPT_DIR}/02_cvae_pretrain.slurm")
        echo "  Submitted: ${JOB_CVAE}"

        echo "[3] VAE pretrain (after rep)"
        JOB_VAE=$(sbatch --parsable --dependency=afterok:${JOB_REP} "${SCRIPT_DIR}/03_vae_pretrain.slurm")
        echo "  Submitted: ${JOB_VAE}"

        echo "[4] Stage2 CVAE (after cvae pretrain)"
        sbatch --dependency=afterok:${JOB_CVAE} "${SCRIPT_DIR}/04_stage2_cvae.slurm"

        echo "[5] Stage2 VAE (after vae pretrain)"
        sbatch --dependency=afterok:${JOB_VAE} "${SCRIPT_DIR}/04_stage2_vae.slurm"

        echo "[6] Stage2 PPO (after data, no stage1 needed)"
        sbatch --dependency=afterok:${JOB_DATA} "${SCRIPT_DIR}/04_stage2_ppo.slurm"

        echo "[7] Stage2 LSTM (after data)"
        sbatch --dependency=afterok:${JOB_DATA} "${SCRIPT_DIR}/04_stage2_lstm.slurm"

        echo "[8] Stage2 GRU (after data)"
        sbatch --dependency=afterok:${JOB_DATA} "${SCRIPT_DIR}/04_stage2_gru.slurm"
        ;;
    *)
        echo "Usage: bash launch_all.sh [0|1|2|all]"
        echo "  0   - Data collection only"
        echo "  1   - Stage1 pretraining"
        echo "  2   - Stage2 PPO training"
        echo "  all - Full pipeline with SLURM dependencies"
        ;;
esac
