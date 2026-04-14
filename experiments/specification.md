# Experiment Specification: CVAE-v6 (Action-Informed Belief) Evaluation

**Created**: 2026-04-12
**Last Updated**: 2026-04-12
**Status**: Planning

---

## 1. Objective

Evaluate the superiority of our main method **CVAE-v6** (Action-Informed Belief, AIB) against multiple baselines. The core contribution is adding a **flow-matching-based action-informed constraint** on top of the Believer baseline. We hypothesize that this constraint either stabilizes the control process or reshapes the belief distribution to be more beneficial for downstream policy learning. This experiment suite aims to provide sufficient empirical evidence.

---

## 2. Experiment Structure (Priority Order)

| Priority | Component | Description |
|----------|-----------|-------------|
| **P0** | Main Experiments | 6 MIKASA-Robo tasks, 5 methods (1 ours + 4 baselines), 3 seeds |
| **P1** | Supplementary: Ablation | 2 ablation variants on MIKASA-Robo tasks |
| **P1** | Supplementary: Sphinx | 5 simple MiniGrid tasks, believer vs CVAE-v6, 3 seeds |
| **P2** | Visualization | t-SNE / quantitative comparison of belief distributions |

---

## 3. Main Experiments (P0)

### 3.1 Tasks (MIKASA-Robo Benchmark)

| # | Environment | Type | Episode Length |
|---|-------------|------|---------------|
| 1 | `RememberColor9-v0` | Remember | Short |
| 2 | `RememberShapeAndColor3x2-v0` | Remember | Short |
| 3 | `RememberShape9-v0` | Remember | Short |
| 4 | `RememberShapeAndColor3x3-v0` | Remember | Short |
| 5 | `InterceptFast-v0` | Intercept | Long |
| 6 | `InterceptMedium-v0` | Intercept | Long |

### 3.2 Methods (5 total)

| # | Method | Script | Requires Stage1 |
|---|--------|--------|-----------------|
| 1 | **CVAE-v6 (Ours)** | `ppo_memtasks_cvae.py` | Yes (rep pretrain + CVAE-v6 pretrain) |
| 2 | PPO (baseline) | `ppo_memtasks.py` | No |
| 3 | PPO + LSTM | `ppo_memtasks_lstm.py` | No |
| 4 | PPO + GRU | `ppo_memtasks_gru.py` | No |
| 5 | PPO + Believer (VAE) | `ppo_memtasks_cvae.py` | Yes (rep pretrain + VAE pretrain) |

### 3.3 Seeds

All experiments use seeds **33, 42, 99**. The same seed must be used consistently from Stage1 (data collection, representation pretraining, VAE/CVAE pretraining) through Stage2 (PPO training).

### 3.4 Unified PPO Parameters

These parameters apply to **all 5 methods** across all tasks:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning-rate` | 1e-4 | Conservative for stability |
| `update-epochs` | 2 | Prevents curve collapse on hard tasks |
| `gae-lambda` | 0.9 | Consistent across prior experiments |
| `gamma` | 0.99 | Conservative (InterceptFast standard) |
| `ent_coef` | 0.001 | Consistent across prior experiments |
| `target-kl` | 0.05 | Early stopping for stability |
| `num-envs` | 256 | Divisible by 32 (num_minibatches) |
| `num-eval-envs` | 16 | Consistent |
| `total-timesteps` | 10,000,000 | Unified |
| `no-finite-horizon-gae` | True | Consistent |
| `num-minibatches` | 32 | Default; 256 % 32 = 0 |

### 3.5 Environment-Specific Parameters

| Task Type | num-steps | num-eval-steps | eval-freq |
|-----------|-----------|----------------|-----------|
| Remember* | 60 | 720 | 48 |
| Intercept* | 90 | 1080 | 25 |

### 3.6 LSTM/GRU-Specific Parameters

| Parameter | Value |
|-----------|-------|
| `lstm-hidden-size` / `gru-hidden-size` | 512 |
| `lstm-num-layers` / `gru-num-layers` | 1 |
| `lstm-dropout` / `gru-dropout` | 0.0 |

Note: LSTM and GRU use the **sequence-based minibatch** strategy (split by environment index, not flat shuffle), consistent with the corrected `ppo_memtasks_gru.py`.

### 3.7 Stage1 Parameters (for CVAE-v6 and Believer VAE)

#### 3.7.1 Data Collection

```
python get_random_datasets_full_state.py \
    --env-id <ENV> \
    --path-to-save-data data \
    --num-episodes 5000 \
    --batch-size 128 \
    --seed <SEED>
```

Data collection uses `mikasa_robo_suite` from the `believer` conda environment.

#### 3.7.2 Representation Pretraining

```
python pretrain_representations_mikasa.py \
    --env <ENV> \
    --model <ENV>_last5 \
    --data-path data/collect_<ENV>.pt \
    --seed <SEED> \
    --epochs 200 \
    --state-sentinel-replace 1.0 \
    --batch-size 500 \
    --latent-dim 16 \
    --beta 0.03 \
    --dynamics-loss-s-coef 0.3 \
    --dynamics-loss-o-coef 0.03 \
    --reward-loss-coef 10
```

#### 3.7.3 CVAE-v6 Pretraining (Our Method)

```
python pretrain_cvae_mikasa.py \
    --env <ENV> \
    --data-path data/collect_<ENV>.pt \
    --model <ENV>_cvae_mikasa_last5_v6 \
    --save-dir storage_cvae \
    --save-interval 50 \
    --seed <SEED> \
    --beta 1 \
    --state-sentinel-replace 1.0 \
    --lambda-action 1.0 \
    --epochs 6000 \
    --lr 0.0003 \
    --latent-dim 32 \
    --batch-size 256 \
    --rep-model-path storage/<ENV>_last5/final_model.pt
```

Algorithm file: `algo_cvae_pretrain_mikasa_v6.py`

#### 3.7.4 VAE Pretraining (Believer Baseline)

```
python pretrain_vae_mikasa.py \
    --env <ENV> \
    --data-path data/collect_<ENV>.pt \
    --model <ENV>_vae_mikasa_last5 \
    --save-dir storage \
    --save-interval 50 \
    --seed <SEED> \
    --beta 1 \
    --state-sentinel-replace 1.0 \
    --epochs 6000 \
    --lr 0.0003 \
    --latent-dim 32 \
    --batch-size 256 \
    --rep-model-path storage/<ENV>_last5/final_model.pt
```

### 3.8 Experiment Scale

- 6 tasks x 5 methods x 3 seeds = **90 Stage2 runs**
- 6 tasks x 2 belief methods (CVAE + VAE) x 3 seeds = **36 Stage1 pretrain runs**
- 6 tasks x 3 seeds = **18 data collection runs** (shared by CVAE and VAE)
- 6 tasks x 3 seeds = **18 representation pretrain runs** (shared by CVAE and VAE)

---

## 4. Supplementary Experiments

### 4.1 Ablation Study (P1)

Purpose: Demonstrate the importance of the **flow-matching-based action-informed** component.

#### 4.1.1 Ablation Variants

| Variant | Description | Algorithm File |
|---------|-------------|----------------|
| **No action head** (`--action-head none`) | Removes action-informed constraint entirely | `algo_cvae_pretrain_mikasa_ablation.py` |
| **MSE inverse dynamics** (`--action-head bc`) | Replaces flow matching with traditional MSE-based behavior cloning | `algo_cvae_pretrain_mikasa_ablation.py` |

#### 4.1.2 Ablation Tasks

Same 6 MIKASA-Robo tasks, same 3 seeds (33, 42, 99), same unified PPO parameters. Only Stage1 CVAE pretraining differs (uses ablation script).

#### 4.1.3 Ablation Scale

- 6 tasks x 2 ablation variants x 3 seeds = **36 additional Stage2 runs**
- 6 tasks x 2 ablation variants x 3 seeds = **36 additional Stage1 CVAE ablation pretrain runs**

### 4.2 Sphinx Supplementary Experiments (P1)

Purpose: Demonstrate generalization on simpler MiniGrid environments from the original Believer/Sphinx framework.

#### 4.2.1 Tasks

| # | Environment |
|---|-------------|
| 1 | `MiniGrid-Genie-8x8-v0` |
| 2 | `MiniGrid-NoisyTV-Genie-8x8-v0` |
| 3 | `MiniGrid-Lying-Genie-8x8-v0` |
| 4 | `MiniGrid-Modified-Cookie-9x9-v0` |
| 5 | `EscapeRoom-v0` |

#### 4.2.2 Methods

- **Believer (VAE baseline)**: `--algo belief_vae`
- **CVAE-v6 (Ours)**: `--algo belief_cvae_v6` (or equivalent)

#### 4.2.3 Parameters

Use the existing Sphinx parameters per environment. **Critical change**: seeds 33, 42, 99 must be used consistently from Stage1 through Stage2 (previously Stage1 used a single seed with multiple Stage2 seeds).

**Standard environments** (Genie, NoisyTV, Lying):

| Parameter | Value |
|-----------|-------|
| `frames` | 5,000,000 |
| `procs` | 32 |
| `frames_per_proc` | 256 |
| `batch-size` | 2,048 |
| `epochs` | 24 |
| `epochs_g` | 8 |
| `lr` | 0.0005 |
| `lr-g` | 0.0003 |
| `entropy-coef` | 0.03 |
| `latent-dim-vae` | 32 |
| `latent-dim-f` | 16 |

**EscapeRoom-v0**:

| Parameter | Value |
|-----------|-------|
| `frames` | 5,000,000 |
| `epochs` | 8 |
| `batch-size-g` | 2,048 |
| `entropy-coef` | 0.01 |
| (other params same as standard) | |

**Modified-Cookie-9x9-v0**:

| Parameter | Value |
|-----------|-------|
| `frames` | 10,000,000 |
| `frames_per_proc` | 512 |
| `batch-size` | 4,096 |
| `batch-size-g` | 4,096 |
| `epochs` | 8 |
| `epochs_g` | 8 |
| `lr` | 0.001 |
| `lr-g` | 0.001 |
| `entropy-coef` | 0.003 |
| `latent-dim-vae` | 64 |
| `discount` | 0.97 |

#### 4.2.4 Sphinx Scale

- 5 tasks x 2 methods x 3 seeds = **30 Stage2 runs**
- 5 tasks x 2 methods x 3 seeds = **30 Stage1 pretrain runs**

---

## 5. Visualization Experiments (P2)

### 5.1 Research Question

> With the AIB method, how closely does the predicted belief state match the ground truth state embeddings? Can you show visualization and quantitative comparison of AIB against Believer along example robot trajectories?

### 5.2 Approach

1. **Qualitative**: t-SNE visualization of belief latent representations vs ground-truth state embeddings along trajectories, comparing CVAE-v6 against Believer VAE.
2. **Quantitative**: Compute distributional distance metrics (e.g., MMD, KL divergence, cosine similarity) between predicted belief and ground-truth embeddings across trajectory steps.

### 5.3 Note on t-SNE

t-SNE is suitable for qualitative visualization but has known limitations (perplexity sensitivity, non-preservation of global structure, non-reproducibility across runs). Recommendations:
- Fix perplexity and random seed for fair comparison
- Supplement with quantitative metrics (MMD or cosine similarity) that do not depend on dimensionality reduction
- Consider UMAP as an alternative that better preserves global structure

---

## 6. Evaluation Protocol

### 6.1 Metrics

All experiments must report the following under **eval mode**:

| Metric | Description |
|--------|-------------|
| `success_once` | Whether the agent succeeded at least once during the episode |
| `success_at_end` | Whether the agent is in a success state at the final timestep |

### 6.2 Reporting Requirements

For each experiment group (task x method):

1. **Learning curves**: Plot `success_once` and `success_at_end` (y-axis) vs `steps` (x-axis) for all methods on the same figure, with clear legend labels.
2. **Quantitative summary**: Average of the **last 3 eval checkpoints** for both `success_once` and `success_at_end`, reported per seed and as mean +/- std across seeds.
3. **Documentation**: Each experiment group must have a report file recording parameters, seeds, paths, and results.

### 6.3 Plotting Standards

- All methods on the same figure for each task
- Mean line with shaded std region across seeds
- Clear legend with method names
- Title includes task name
- X-axis: environment steps
- Y-axis: success rate [0, 1]

---

## 7. Naming Conventions

### 7.1 Experiment Names (exp-name)

Format: `<method>-<task_short>-<seed>`

| Method | Prefix |
|--------|--------|
| CVAE-v6 (Ours) | `cvae-v6` |
| PPO | `ppo-baseline` |
| PPO + LSTM | `ppo-lstm` |
| PPO + GRU | `ppo-gru` |
| Believer (VAE) | `ppo-vae` |
| Ablation: no action head | `cvae-ablation-none` |
| Ablation: MSE inverse dynamics | `cvae-ablation-mse` |

Example: `cvae-v6-color9-33`, `ppo-lstm-interceptfast-42`

### 7.2 Stage1 Model Naming

- Representation: `storage/<ENV>_last5_s<SEED>/final_model.pt`
- CVAE-v6: `storage_cvae/<ENV>_cvae_v6_s<SEED>/final_vae_model.pt`
- VAE: `storage/<ENV>_vae_s<SEED>/final_vae_model.pt`

### 7.3 Output Directory Structure

```
checkpoints/
  <ENV>/
    <method>-<seed>/
      <timestamp>/
        ckpt_*.pt
        final_ckpt.pt
        eval_results.csv
        tensorboard/
```

---

## 8. Reproducibility Requirements

Every experiment must provide:

1. **Exact command** or **SLURM script** used to launch the run
2. **Seed** used (consistent across Stage1 and Stage2)
3. **Checkpoint paths** for Stage1 models consumed by Stage2
4. **Git commit hash** of the codebase at the time of the experiment
5. **Conda environment** name and key package versions

---

## 9. Additional Constraints

1. **No cherry-picking**: All 3 seeds must be run and reported. Failed runs must be documented with error logs.
2. **Consistent evaluation**: All methods for the same task must use the same eval frequency and eval episode count.
3. **GPU reproducibility**: Set `torch.backends.cudnn.deterministic = True` where feasible, but document if not achievable.
4. **Checkpoint saving**: Save model checkpoints at each eval point (`--save-model`). Keep at minimum the final checkpoint and the best-performing checkpoint.
5. **Logging**: All runs must log to TensorBoard and CSV. CSV files are the primary source for final result tables.
6. **Concurrent runs**: When submitting SLURM jobs, use `--array=0-2` with `SEEDS=(33 42 99)` to run 3 seeds concurrently where possible.
7. **Model name collision prevention**: Stage1 pretrain scripts (`pretrain_representations_mikasa.py`, `pretrain_cvae_mikasa_v6.py`, `pretrain_vae_mikasa.py`) auto-resume from `status.pt` if the output directory (`--model`) already exists. **Before launching any experiment, verify that the `--model` name does not collide with an existing directory in `storage/` or `storage_cvae/`.** If a directory with the same name exists from a previous run, the script will load the old checkpoint and continue training instead of starting fresh — silently producing incorrect results. When re-running an experiment from scratch, either delete the old directory or use a distinct `--model` name.

---

## 10. Environment Setup

| Component | Conda Env | Location |
|-----------|-----------|----------|
| MIKASA-Robo Stage2 | `mikasa` | `/local/s4176650/MIKASA-Robo` |
| Sphinx Stage1 (MIKASA tasks) | `believer` | `/local/s4176650/sphinx` |
| Sphinx Stage1+2 (MiniGrid tasks) | `believer` | `/local/s4176650/sphinx` |
| Data collection (MIKASA tasks) | `believer` | `/local/s4176650/sphinx` |
| `mikasa_robo_suite` package | installed in `believer` | `/data/s4176650/conda_envs/believer/lib/python3.9/site-packages/mikasa_robo_suite/` |

---

## 11. Summary Table

| Component | Tasks | Methods | Seeds | Total Runs (Stage2) |
|-----------|-------|---------|-------|---------------------|
| Main | 6 | 5 | 3 | 90 |
| Ablation | 6 | 2 | 3 | 36 |
| Sphinx | 5 | 2 | 3 | 30 |
| **Total** | | | | **156** |
