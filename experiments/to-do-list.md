# To-Do List

**Last Updated**: 2026-04-14

---

## P0: Main Experiments (MIKASA-Robo)

### Data Collection & Stage1

- [ ] Collect random datasets for 6 tasks (seed=42)
  - [ ] RememberColor9-v0 (~15G)
  - [ ] RememberShapeAndColor3x2-v0 (~15G)
  - [ ] RememberShape9-v0 (~15G)
  - [ ] RememberShapeAndColor3x3-v0 (~15G)
  - [ ] InterceptFast-v0 (~22G)
  - [ ] InterceptMedium-v0 (~22G)

- [ ] Representation pretraining for 6 tasks x 3 seeds (18 runs)
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] CVAE-v6 pretraining for 6 tasks x 3 seeds (18 runs)
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] VAE (Believer) pretraining for 6 tasks x 3 seeds (18 runs)
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

### Stage2: PPO Training (90 runs)

- [ ] CVAE-v6 (Ours): 6 tasks x 3 seeds = 18 runs
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] PPO baseline: 6 tasks x 3 seeds = 18 runs
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] PPO + LSTM: 6 tasks x 3 seeds = 18 runs
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] PPO + GRU: 6 tasks x 3 seeds = 18 runs
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

- [ ] PPO + Believer (VAE): 6 tasks x 3 seeds = 18 runs
  - [ ] RememberColor9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x2-v0: seeds 33, 42, 99
  - [ ] RememberShape9-v0: seeds 33, 42, 99
  - [ ] RememberShapeAndColor3x3-v0: seeds 33, 42, 99
  - [ ] InterceptFast-v0: seeds 33, 42, 99
  - [ ] InterceptMedium-v0: seeds 33, 42, 99

### Results & Plotting

- [ ] Generate learning curve plots for each task (6 plots, each with 5 methods)
- [ ] Compute last-3-eval quantitative tables for each task
- [ ] Compile main results summary table

---

## P1: Ablation Experiments

### Stage1: Ablation Pretraining

- [ ] Ablation "none" (no action head) CVAE pretraining: 6 tasks x 3 seeds = 18 runs
- [ ] Ablation "mse" (MSE inverse dynamics) CVAE pretraining: 6 tasks x 3 seeds = 18 runs

### Stage2: Ablation PPO Training (36 runs)

- [ ] Ablation "none": 6 tasks x 3 seeds = 18 runs
- [ ] Ablation "mse": 6 tasks x 3 seeds = 18 runs

### Results

- [ ] Generate ablation learning curve plots (CVAE-v6 vs no-action-head vs MSE)
- [ ] Compute ablation quantitative tables

---

## P1: Sphinx Supplementary Experiments

### Stage1

- [ ] Data collection for 5 MiniGrid tasks (if not already done)
- [ ] Representation pretraining for 5 tasks x 3 seeds
- [ ] VAE pretraining for 5 tasks x 3 seeds
- [ ] CVAE-v6 pretraining for 5 tasks x 3 seeds

### Stage2

- [ ] Believer (VAE): 5 tasks x 3 seeds = 15 runs
- [ ] CVAE-v6 (Ours): 5 tasks x 3 seeds = 15 runs

### Results

- [ ] Generate learning curve plots for 5 Sphinx tasks
- [ ] Compute quantitative tables

---

## P2: Visualization Experiments

- [ ] Collect trajectory data with trained CVAE-v6 and Believer agents
- [ ] Extract belief latent representations and ground-truth state embeddings
- [ ] Generate t-SNE visualizations
- [ ] Compute quantitative distributional metrics (MMD / cosine similarity)
- [ ] Create comparison figure panels

---

## Infrastructure & Preparation

- [x] Update MIKASA-Robo codebase to latest
- [x] Fix GRU minibatch logic (revert to sequence-based, consistent with LSTM)
- [x] Confirm CVAE-v6 uses `algo_cvae_pretrain_mikasa_v6.py`
- [x] Confirm `mikasa_robo_suite` installed in `believer` conda env
- [x] Copy and organize reference SLURM scripts
- [x] Write SLURM scripts for all main experiments
- [ ] Write SLURM scripts for ablation experiments
- [ ] Write SLURM scripts for Sphinx supplementary experiments
- [ ] Write plotting and evaluation scripts
- [ ] Record git commit hash before launching experiments
