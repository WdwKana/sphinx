#!/usr/bin/env python3
"""
Latent-action structure analysis for Mikasa CVAE v6 checkpoints.

What this script does:
1) Reuses v6 pretraining data flow:
   - load collect data (.pt)
   - transpose to (T, N, ...)
   - apply optional state sentinel replacement
   - encode state with representation model
2) Reuses v6 latent extraction logic:
   - history update via BeliefVAEModel(obs, memory)
   - posterior mean via encoder_dist(...): mu_t
   - transition features:
       pair  = [mu_{t-1}, mu_t]
       delta = mu_t - mu_{t-1}
     with label action_{t-1}
3) Produces:
   - linear ridge probe metrics (R2 / nMSE) on held-out episodes
   - t-SNE visualization (fallback to PCA if sklearn TSNE unavailable)

Designed for rebuttal analysis without retraining belief models.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_cvae_mikasa import BeliefVAEModel
from model_f_mikasa import RepresentationModelMikasa


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_model_specs(paths_csv: str, labels_csv: str) -> List[Tuple[str, str]]:
    paths = parse_csv_list(paths_csv)
    if not paths:
        raise ValueError("No --belief-paths provided.")

    labels = parse_csv_list(labels_csv) if labels_csv else []
    if labels and len(labels) != len(paths):
        raise ValueError(
            f"--belief-labels length ({len(labels)}) must match --belief-paths length ({len(paths)})."
        )

    specs: List[Tuple[str, str]] = []
    for i, p in enumerate(paths):
        label = labels[i] if labels else Path(p).parent.name
        specs.append((label, p))
    return specs


def maybe_dict_get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_collect_data(
    *,
    data_path: str,
    seed: int,
    num_episodes: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(data_path, map_location="cpu")

    # Same keys used by pretrain_cvae_mikasa_v6.py
    obss = data["obss"]      # (N, T, H, W, C)
    states = data["states"]  # (N, T, D)
    actions = data["actions"]
    masks = data["masks"]

    if num_episodes is not None:
        total_eps = int(masks.shape[0])
        k = int(num_episodes)
        if k <= 0:
            raise ValueError("--num-episodes must be > 0")
        k = min(k, total_eps)
        g = torch.Generator()
        g.manual_seed(seed)
        ep_ids = torch.randperm(total_eps, generator=g)[:k]
        ep_ids, _ = torch.sort(ep_ids)

        def _slice_episode_dim(tensor):
            return tensor[ep_ids] if torch.is_tensor(tensor) and tensor.shape[0] == total_eps else tensor

        obss = _slice_episode_dim(obss)
        states = _slice_episode_dim(states)
        actions = _slice_episode_dim(actions)
        masks = _slice_episode_dim(masks)

    # Same transpose as pretrain_cvae_mikasa_v6.py
    obss = obss.transpose(0, 1).contiguous()    # (T, N, H, W, C)
    states = states.transpose(0, 1).contiguous()  # (T, N, D)
    actions = actions.transpose(0, 1).contiguous()
    masks = masks.transpose(0, 1).contiguous()
    return obss, states, actions, masks


def resolve_rep_sentinel_args(
    *,
    rep_checkpoint: Dict,
    state_sentinel: float,
    state_sentinel_replace: float | None,
) -> Tuple[float, float | None]:
    inherited_replace = None
    inherited_sentinel = None
    if isinstance(rep_checkpoint, dict):
        if "state_sentinel_replace" in rep_checkpoint:
            inherited_replace = rep_checkpoint.get("state_sentinel_replace")
        else:
            rep_args = rep_checkpoint.get("args", {})
            if isinstance(rep_args, dict) and "state_sentinel_replace" in rep_args:
                inherited_replace = rep_args.get("state_sentinel_replace")

        if "state_sentinel" in rep_checkpoint:
            inherited_sentinel = rep_checkpoint.get("state_sentinel")
        else:
            rep_args = rep_checkpoint.get("args", {})
            if isinstance(rep_args, dict) and "state_sentinel" in rep_args:
                inherited_sentinel = rep_args.get("state_sentinel")

    if state_sentinel_replace is None and inherited_replace is not None:
        state_sentinel_replace = float(inherited_replace)
    if inherited_sentinel is not None:
        state_sentinel = float(inherited_sentinel)
    return float(state_sentinel), state_sentinel_replace


def load_rep_model(
    *,
    rep_model_path: str,
    obs_shape: Tuple[int, ...],
    raw_state_dim: int,
    action_dim: int,
    device: torch.device,
) -> Tuple[RepresentationModelMikasa, Dict]:
    rep_checkpoint = torch.load(rep_model_path, map_location=device)
    rep_args = rep_checkpoint.get("args", {}) if isinstance(rep_checkpoint, dict) else {}
    rep_latent_dim = int(maybe_dict_get(rep_args, "latent_dim", 16))

    rep_model = RepresentationModelMikasa(
        obs_shape=obs_shape,
        state_dim=raw_state_dim,
        action_dim=action_dim,
        latent_dim=rep_latent_dim,
    ).to(device)
    rep_model.load_state_dict(rep_checkpoint["model_state"])
    rep_model.eval()
    for p in rep_model.parameters():
        p.requires_grad = False
    return rep_model, rep_checkpoint


def load_belief_model_v6(
    *,
    belief_path: str,
    default_obs_shape: Tuple[int, ...],
    device: torch.device,
) -> Tuple[BeliefVAEModel, Dict[str, int | Tuple[int, ...]]]:
    """
    Loads final belief checkpoint with robust shape inference.
    Compatible with v6 final_vae_model.pt format.
    """
    ckpt = torch.load(belief_path, map_location=device)
    model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    phi_dim = None
    if isinstance(ckpt, dict):
        phi_dim = ckpt.get("phi_dim")
        if phi_dim is None:
            phi_dim = ckpt.get("state_dim")
    if phi_dim is None and "vae_decoder.4.weight" in model_state:
        # output is 2 * phi_dim
        phi_dim = int(model_state["vae_decoder.4.weight"].shape[0] // 2)
    if phi_dim is None:
        raise ValueError(f"Cannot infer phi_dim from checkpoint: {belief_path}")

    latent_dim = None
    if isinstance(ckpt, dict):
        latent_dim = ckpt.get("latent_dim")
        if latent_dim is None:
            ckpt_args = ckpt.get("args", {})
            if isinstance(ckpt_args, dict):
                latent_dim = ckpt_args.get("latent_dim")
    if latent_dim is None and "vae_encoder.4.weight" in model_state:
        latent_dim = int(model_state["vae_encoder.4.weight"].shape[0] // 2)
    if latent_dim is None:
        raise ValueError(f"Cannot infer latent_dim from checkpoint: {belief_path}")

    obs_shape = default_obs_shape
    if isinstance(ckpt, dict) and "obs_shape" in ckpt:
        obs_shape = tuple(ckpt["obs_shape"])

    belief_model = BeliefVAEModel(
        obs_space=obs_shape,
        state_dim=int(phi_dim),
        latent_dim=int(latent_dim),
    ).to(device)
    belief_model.load_state_dict(model_state)
    belief_model.eval()
    for p in belief_model.parameters():
        p.requires_grad = False
    meta = {
        "phi_dim": int(phi_dim),
        "latent_dim": int(latent_dim),
        "obs_shape": tuple(obs_shape),
    }
    return belief_model, meta


def split_episode_indices(num_episodes: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_episodes < 2:
        raise ValueError(f"Need at least 2 episodes, got {num_episodes}")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in (0, 1), got {test_ratio}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    n_test = int(round(num_episodes * test_ratio))
    n_test = min(max(1, n_test), num_episodes - 1)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


@dataclass
class TransitionFeatures:
    pair: np.ndarray   # [mu_{t-1}, mu_t]
    delta: np.ndarray  # mu_t - mu_{t-1}
    action: np.ndarray # action_{t-1}


def extract_transition_features_v6(
    *,
    obss: torch.Tensor,   # (T, N, H, W, C) on CPU
    states: torch.Tensor, # (T, N, D) on CPU
    actions: torch.Tensor,
    masks: torch.Tensor,
    episode_indices: np.ndarray,
    rep_model: RepresentationModelMikasa,
    belief_model: BeliefVAEModel,
    device: torch.device,
    history_recurrence: int,
) -> TransitionFeatures:
    idx = torch.as_tensor(episode_indices, dtype=torch.long)
    max_steps = int(masks.shape[0])
    num_eps = int(idx.shape[0])

    pair_feats: List[torch.Tensor] = []
    delta_feats: List[torch.Tensor] = []
    action_labels: List[torch.Tensor] = []

    memory = torch.zeros((num_eps, belief_model.history_model.memory_size), device=device)
    prev_mu = None
    prev_mask = None
    prev_action = None

    with torch.no_grad():
        for step in range(max_steps):
            mask_cpu = masks[step].index_select(0, idx)
            if mask_cpu.sum().item() <= 0:
                break

            # Per-step transfer to device to avoid loading full dataset on GPU.
            obs_t = obss[step].index_select(0, idx).to(device)
            state_t = states[step].index_select(0, idx).to(device)
            action_t = actions[step].index_select(0, idx).to(device)
            mask_t = mask_cpu.to(device).float()

            # Same representation path as v6 pretraining.
            rep_encoder_mean, _ = rep_model.encode_state(state_t)
            state_features = rep_encoder_mean

            # Same recurrent update as v6.
            if step % history_recurrence == 0:
                history_encoding, memory = belief_model(obs_t, memory.detach() * mask_t.unsqueeze(1))
            else:
                history_encoding, memory = belief_model(obs_t, memory * mask_t.unsqueeze(1))

            # Same posterior mean used as conditioning in v6 FM.
            encoder_mean, _ = belief_model.encoder_dist(state_features, history_encoding)
            mu_t = encoder_mean

            if prev_mu is not None and prev_action is not None and prev_mask is not None:
                valid = (prev_mask > 0) & (mask_t > 0)
                if valid.any():
                    prev_valid = prev_mu[valid]
                    curr_valid = mu_t[valid]
                    pair_feats.append(torch.cat([prev_valid, curr_valid], dim=1).cpu())
                    delta_feats.append((curr_valid - prev_valid).cpu())

                    act = prev_action[valid]
                    if act.dim() == 1:
                        act = act.unsqueeze(1)
                    else:
                        act = act.flatten(start_dim=1)
                    action_labels.append(act.float().cpu())

            prev_mu = mu_t
            prev_mask = mask_t
            prev_action = action_t

    if not pair_feats:
        raise RuntimeError("No valid transitions extracted. Check masks/data split.")

    pair = torch.cat(pair_feats, dim=0).numpy()
    delta = torch.cat(delta_feats, dim=0).numpy()
    action = torch.cat(action_labels, dim=0).numpy()
    return TransitionFeatures(pair=pair, delta=delta, action=action)


def fit_ridge_and_eval(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    eps = 1e-8

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + eps
    xs_train = (x_train - x_mean) / x_std
    xs_test = (x_test - x_mean) / x_std

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + eps
    ys_train = (y_train - y_mean) / y_std

    # Add bias term
    ones_train = np.ones((xs_train.shape[0], 1), dtype=xs_train.dtype)
    ones_test = np.ones((xs_test.shape[0], 1), dtype=xs_test.dtype)
    xb_train = np.concatenate([ones_train, xs_train], axis=1)
    xb_test = np.concatenate([ones_test, xs_test], axis=1)

    d = xb_train.shape[1]
    reg = np.eye(d, dtype=xb_train.dtype) * float(alpha)
    reg[0, 0] = 0.0  # do not regularize bias
    w = np.linalg.solve(xb_train.T @ xb_train + reg, xb_train.T @ ys_train)
    ys_pred_test = xb_test @ w
    y_pred_test = ys_pred_test * y_std + y_mean

    mse_dim = np.mean((y_test - y_pred_test) ** 2, axis=0)
    var_train_dim = np.var(y_train, axis=0) + eps
    nmse_dim = mse_dim / var_train_dim
    nmse_mean = float(np.mean(nmse_dim))

    ss_res = np.sum((y_test - y_pred_test) ** 2, axis=0)
    ss_tot = np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2, axis=0) + eps
    r2_dim = 1.0 - (ss_res / ss_tot)
    r2_mean = float(np.mean(r2_dim))

    return {
        "r2_mean": r2_mean,
        "nmse_mean": nmse_mean,
        "mse_mean": float(np.mean(mse_dim)),
        "n_outputs": float(y_train.shape[1]),
    }


def pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    if x0.shape[1] == 1:
        return np.concatenate([x0, np.zeros_like(x0)], axis=1)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comp = vt[:2].T
    return x0 @ comp


def embed_2d_tsne_or_pca(
    x: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> Tuple[np.ndarray, str]:
    if x.shape[0] < 5:
        return pca_2d(x), "pca_small_n"

    try:
        from sklearn.manifold import TSNE

        p = float(perplexity)
        p = min(p, max(5.0, (x.shape[0] - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=p,
            n_iter=max_iter,
            random_state=seed,
        )
        return tsne.fit_transform(x), "tsne"
    except Exception:
        return pca_2d(x), "pca_fallback"


def sample_for_vis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=max_points, replace=False)
    return x[sel], y[sel]


def action_color_value(y: np.ndarray) -> Tuple[np.ndarray, str]:
    if y.ndim == 1 or y.shape[1] == 1:
        return y.reshape(-1), "action[0]"
    return np.linalg.norm(y, axis=1), "||action||_2"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Environment name (for report naming).")
    parser.add_argument("--data-path", required=True, help="Path to collect_*.pt dataset.")
    parser.add_argument(
        "--belief-paths",
        required=True,
        help="Comma-separated belief checkpoint paths (final_vae_model.pt), e.g. none,v6_001,v6_01,v6_1.",
    )
    parser.add_argument(
        "--belief-labels",
        default="",
        help="Optional comma-separated labels aligned with --belief-paths.",
    )
    parser.add_argument(
        "--rep-model-path",
        default=None,
        help="Representation model path. Default: storage/{env}_last5/final_model.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="/local/s4176650/MIKASA-Robo/final_results/latent_action_structure",
        help="Directory to write figures/reports.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--state-sentinel", type=float, default=1000.0)
    parser.add_argument("--state-sentinel-replace", type=float, default=None)
    parser.add_argument("--history-recurrence", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-tsne-points", type=int, default=12000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    args = parser.parse_args()

    seed_all(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    print(f"[info] device={device}")
    print(f"[info] loading data: {args.data_path}")
    obss, states, actions, masks = load_collect_data(
        data_path=args.data_path,
        seed=args.seed,
        num_episodes=args.num_episodes,
    )
    t_steps, n_episodes = int(masks.shape[0]), int(masks.shape[1])
    print(f"[info] transposed data shapes: obss={tuple(obss.shape)}, states={tuple(states.shape)}, actions={tuple(actions.shape)}, masks={tuple(masks.shape)}")

    obs_shape = tuple(obss.shape[2:])
    raw_state_dim = int(states.shape[2])
    action_dim = int(actions.shape[2]) if actions.dim() > 2 else 1

    default_rep_path = f"storage/{args.env}_last5/final_model.pt"
    rep_model_path = args.rep_model_path or default_rep_path
    if not os.path.exists(rep_model_path):
        raise FileNotFoundError(f"Representation model not found at {rep_model_path}")
    print(f"[info] loading rep model: {rep_model_path}")
    rep_model, rep_ckpt = load_rep_model(
        rep_model_path=rep_model_path,
        obs_shape=obs_shape,
        raw_state_dim=raw_state_dim,
        action_dim=action_dim,
        device=device,
    )

    # Reuse v6 sentinel inheritance logic.
    state_sentinel, state_sentinel_replace = resolve_rep_sentinel_args(
        rep_checkpoint=rep_ckpt,
        state_sentinel=args.state_sentinel,
        state_sentinel_replace=args.state_sentinel_replace,
    )
    if state_sentinel_replace is not None:
        print(f"[info] state sentinel replace: {state_sentinel} -> {state_sentinel_replace}")
        states = torch.where(states == float(state_sentinel), torch.tensor(float(state_sentinel_replace), dtype=states.dtype), states)

    train_idx, test_idx = split_episode_indices(
        num_episodes=n_episodes,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[info] split episodes: train={len(train_idx)}, test={len(test_idx)}")

    model_specs = parse_model_specs(args.belief_paths, args.belief_labels)
    print(f"[info] models: {[label for label, _ in model_specs]}")

    metrics_rows: List[Dict[str, object]] = []
    vis_pack: List[Dict[str, object]] = []

    for i, (label, belief_path) in enumerate(model_specs):
        if not os.path.exists(belief_path):
            raise FileNotFoundError(f"Belief checkpoint not found: {belief_path}")

        print(f"[info] [{i+1}/{len(model_specs)}] loading belief: {label} -> {belief_path}")
        belief_model, meta = load_belief_model_v6(
            belief_path=belief_path,
            default_obs_shape=obs_shape,
            device=device,
        )

        print(f"[info] extracting transitions (train): {label}")
        train_feat = extract_transition_features_v6(
            obss=obss,
            states=states,
            actions=actions,
            masks=masks,
            episode_indices=train_idx,
            rep_model=rep_model,
            belief_model=belief_model,
            device=device,
            history_recurrence=args.history_recurrence,
        )
        print(f"[info] extracting transitions (test): {label}")
        test_feat = extract_transition_features_v6(
            obss=obss,
            states=states,
            actions=actions,
            masks=masks,
            episode_indices=test_idx,
            rep_model=rep_model,
            belief_model=belief_model,
            device=device,
            history_recurrence=args.history_recurrence,
        )

        probe = fit_ridge_and_eval(
            x_train=train_feat.pair,
            y_train=train_feat.action,
            x_test=test_feat.pair,
            y_test=test_feat.action,
            alpha=args.ridge_alpha,
        )

        metrics_rows.append(
            {
                "label": label,
                "belief_path": belief_path,
                "phi_dim": meta["phi_dim"],
                "latent_dim": meta["latent_dim"],
                "n_train_transitions": int(train_feat.pair.shape[0]),
                "n_test_transitions": int(test_feat.pair.shape[0]),
                "action_dim": int(test_feat.action.shape[1]),
                "r2_mean": probe["r2_mean"],
                "nmse_mean": probe["nmse_mean"],
                "mse_mean": probe["mse_mean"],
            }
        )

        vis_x, vis_y = sample_for_vis(
            test_feat.delta,
            test_feat.action,
            max_points=args.max_tsne_points,
            seed=args.seed + i,
        )
        vis_pack.append(
            {
                "label": label,
                "x": vis_x,
                "y": vis_y,
                "r2_mean": probe["r2_mean"],
                "nmse_mean": probe["nmse_mean"],
            }
        )

        del belief_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save metrics CSV
    csv_path = os.path.join(args.out_dir, f"{args.env}__latent_action_probe_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "belief_path",
                "phi_dim",
                "latent_dim",
                "n_train_transitions",
                "n_test_transitions",
                "action_dim",
                "r2_mean",
                "nmse_mean",
                "mse_mean",
            ]
        )
        for row in metrics_rows:
            w.writerow(
                [
                    row["label"],
                    row["belief_path"],
                    row["phi_dim"],
                    row["latent_dim"],
                    row["n_train_transitions"],
                    row["n_test_transitions"],
                    row["action_dim"],
                    f"{row['r2_mean']:.8f}",
                    f"{row['nmse_mean']:.8f}",
                    f"{row['mse_mean']:.8f}",
                ]
            )

    # Build t-SNE/PCA panel figure
    n = len(vis_pack)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    embed_methods: Dict[str, str] = {}
    for i, item in enumerate(vis_pack):
        ax = axes[i]
        emb, method = embed_2d_tsne_or_pca(
            item["x"],
            seed=args.seed + 31 * i,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_max_iter,
        )
        embed_methods[item["label"]] = method
        c, c_name = action_color_value(item["y"])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis", s=6, alpha=0.7, linewidths=0.0)
        ax.set_title(f"{item['label']} | R2={item['r2_mean']:.3f}, nMSE={item['nmse_mean']:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(c_name)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{args.env}: latent transition structure (delta=mu_t-mu_(t-1))", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = os.path.join(args.out_dir, f"{args.env}__latent_action_structure__tsne_panel.png")
    fig.savefig(fig_path, dpi=250)
    plt.close(fig)

    # Save text report
    rep_path = os.path.join(args.out_dir, f"{args.env}__latent_action_structure__report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"env: {args.env}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"rep_model_path: {rep_model_path}\n")
        f.write(f"out_dir: {args.out_dir}\n")
        f.write(f"device: {device}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_episodes_after_subset: {n_episodes}\n")
        f.write(f"train_episodes: {len(train_idx)}\n")
        f.write(f"test_episodes: {len(test_idx)}\n")
        f.write(f"state_sentinel: {state_sentinel}\n")
        f.write(f"state_sentinel_replace: {state_sentinel_replace}\n")
        f.write(f"history_recurrence: {args.history_recurrence}\n")
        f.write(f"ridge_alpha: {args.ridge_alpha}\n")
        f.write(f"max_tsne_points: {args.max_tsne_points}\n")
        f.write(f"tsne_perplexity: {args.tsne_perplexity}\n")
        f.write(f"tsne_max_iter: {args.tsne_max_iter}\n")
        f.write(f"python_cmd: {' '.join(sys.argv)}\n")
        f.write("\n## Embedding method per model\n")
        for k, v in embed_methods.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Metrics\n")
        for row in metrics_rows:
            f.write(
                f"- {row['label']}: R2={row['r2_mean']:.6f}, "
                f"nMSE={row['nmse_mean']:.6f}, "
                f"n_train={row['n_train_transitions']}, n_test={row['n_test_transitions']}, "
                f"belief_path={row['belief_path']}\n"
            )

    print(f"[done] metrics_csv: {csv_path}")
    print(f"[done] figure: {fig_path}")
    print(f"[done] report: {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
