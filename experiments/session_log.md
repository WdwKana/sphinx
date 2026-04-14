# Session Log

Records of completed work, in reverse chronological order.

---

## 2026-04-12 — Experiment Planning & Infrastructure Setup

### Completed Tasks

1. **Updated MIKASA-Robo codebase**
   - Pulled latest from `https://github.com/WdwKana/MIKASA-Robo.git`
   - Commit: `f8c32d2` -> `b6d03af` (fast-forward, 162 files changed)
   - Resolved conflict: backed up local `baselines/ppo/ppo_memtasks_gru.py` before merge

2. **Fixed GRU minibatch logic**
   - **Problem**: The modified `ppo_memtasks_gru.py` used flat-shuffle minibatches with per-step saved hidden states. This broke the done-reset semantics in `get_states()` because unrelated samples from different envs/timesteps were mixed together.
   - **Solution**: Reverted to `.bak` version which uses sequence-based minibatch (split by environment index), consistent with `ppo_memtasks_lstm.py`.
   - **Action**: Deleted the broken current version; renamed `.bak` to `ppo_memtasks_gru.py`.

3. **Analyzed Stage2 parameter differences across tasks**
   - Identified two parameter groups: Remember-type (short episodes) and Intercept-type (long episodes)
   - Key PPO differences: `update-epochs` (2 vs 4), `gamma` (0.95 vs 0.99), `num-envs` (200 vs 256)
   - Identified `num_envs % num_minibatches != 0` issue when using LSTM with `num-envs=200`

4. **Established unified PPO parameters**
   - `gamma=0.99`, `update-epochs=2`, `num-envs=256`, `learning-rate=1e-4`, `target-kl=0.05`, `total-timesteps=10M`
   - Conservative settings based on hard tasks (InterceptFast/RememberColor9) to ensure stability

5. **Organized reference SLURM scripts**
   - Copied from `~/mikasa/` to `/local/s4176650/sphinx/slurm_script/` with categorized subdirectories:
     - `stage1_pretrain/`, `stage1_vae/`, `stage1_cvae/`
     - `stage2_belief_ppo/`, `stage2_belief_cvae/`
     - `data_collect/`

6. **Confirmed infrastructure**
   - `mikasa_robo_suite 0.0.5` installed in `believer` conda env at `/data/s4176650/conda_envs/believer/`
   - CVAE uses `algo_cvae_pretrain_mikasa_v6.py` (verified)
   - Ablation code exists: `algo_cvae_pretrain_mikasa_ablation.py` with `--action-head none|bc` flags

7. **Created experiment specification document**
   - `experiments/specification.md`: Full experiment plan covering main experiments, ablation, Sphinx supplementary, and visualization
   - `experiments/to-do-list.md`: Detailed checklist of all pending work
   - `experiments/session_log.md`: This file

8. **Wrote main-experiment SLURM scripts**
   - Location: `experiments/scripts/main/`
   - `_common.sh`: Centralized config (tasks, seeds, all hyperparams). The `init_conda()` helper wraps `source ~/.bashrc` with `set +e` / `set -e` so that a non-fatal D-Bus error on compute nodes (`Failed to connect to bus: No such file or directory`) does not kill the job before Python starts.
   - `00_collect_data.slurm`: Data collection (array=0-5)
   - `01_rep_pretrain.slurm`: Representation pretraining (array=0-17, 6 tasks x 3 seeds)
   - `02_cvae_pretrain.slurm`: CVAE-v6 pretraining (array=0-17, depends on rep)
   - `03_vae_pretrain.slurm`: VAE pretraining (array=0-17, depends on rep)
   - `04_stage2_{cvae,vae,ppo,lstm,gru}.slurm`: 5 Stage2 methods (array=0-17 each)
   - `launch_all.sh`: Orchestrator with SLURM dependency chains

### Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Unified `gamma=0.99` across all tasks | More conservative; prevents curve collapse |
| Unified `update-epochs=2` | Slower, more stable updates |
| Unified `num-envs=256` | Must be divisible by `num_minibatches=32` |
| Seeds 33, 42, 99 used end-to-end | Same seed from data collection through Stage2 |
| GRU reverted to sequence-based minibatch | Consistency with LSTM; correctness of done-reset logic |
| CVAE/VAE pretrain epochs = 6000 | Empirically sufficient for convergence; avoids KL collapse seen in very long runs |

### Files Modified

| File | Action |
|------|--------|
| `MIKASA-Robo/baselines/ppo/ppo_memtasks_gru.py` | Replaced with .bak (sequence-based minibatch) |
| `MIKASA-Robo/baselines/ppo/ppo_memtasks_gru.py.bak` | Deleted (renamed to replace current) |

### Open Questions

- Sphinx CVAE-v6 integration: need to confirm how `algo_cvae_pretrain_mikasa_v6.py` integrates with Sphinx's `train.py` for Stage2 (the `--algo` flag)
- Visualization: t-SNE vs UMAP decision pending; recommend supplementing with quantitative metrics regardless
