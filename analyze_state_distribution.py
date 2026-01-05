import argparse
import math
import os
from typing import Dict, List, Tuple

import torch


def _parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip() != ""]


def _format_float(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax >= 1e6 or ax < 1e-4):
        return f"{x:.3e}"
    return f"{x:.6f}"


def _summarize_tensor_1d(x: torch.Tensor, quantiles: torch.Tensor) -> Dict[str, torch.Tensor]:
    # x: (N,)
    out: Dict[str, torch.Tensor] = {}
    out["count"] = torch.tensor([x.numel()], dtype=torch.int64)
    out["nan"] = torch.isnan(x).sum().to(torch.int64)
    out["inf"] = torch.isinf(x).sum().to(torch.int64)
    finite = x[torch.isfinite(x)]
    out["finite_count"] = torch.tensor([finite.numel()], dtype=torch.int64)
    if finite.numel() == 0:
        out["mean"] = torch.tensor([float("nan")])
        out["std"] = torch.tensor([float("nan")])
        out["min"] = torch.tensor([float("nan")])
        out["max"] = torch.tensor([float("nan")])
        out["quantiles"] = torch.full((quantiles.numel(),), float("nan"))
        return out
    out["mean"] = finite.mean()
    out["std"] = finite.std(unbiased=False)
    out["min"] = finite.min()
    out["max"] = finite.max()
    out["quantiles"] = torch.quantile(finite, quantiles)
    return out


def _idx_to_episode_step(flat_idx: torch.Tensor, episode_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ep = flat_idx // episode_len
    t = flat_idx % episode_len
    return ep, t


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze distribution of `states` in a Sphinx-format .pt dataset.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/collect_RememberShapeAndColor3x2-v0.pt",
        help="Path to dataset .pt (expects keys: states, masks).",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0,0.001,0.01,0.05,0.5,0.95,0.99,0.999,1",
        help="Comma-separated quantiles to compute.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many global extreme values (by abs) to print with indices.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,10,100,1000,10000",
        help="Comma-separated |x| thresholds to count per-dimension.",
    )
    parser.add_argument(
        "--sentinels",
        type=str,
        default="1000,4242424242",
        help="Comma-separated sentinel values to count (exact match after casting).",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading: {data_path}")
    data = torch.load(data_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected torch.load(...) to return dict, got {type(data)}")

    print("Keys:", sorted(list(data.keys())))
    if "states" not in data:
        raise KeyError("Dataset does not contain key 'states'")
    states = data["states"]
    masks = data.get("masks", None)

    if not torch.is_tensor(states):
        raise TypeError(f"'states' must be a torch.Tensor, got {type(states)}")
    if states.ndim != 3:
        raise ValueError(f"Expected states shape (N, T, D), got {tuple(states.shape)}")
    n_ep, ep_len, d = states.shape
    print(f"states: shape={tuple(states.shape)} dtype={states.dtype} device={states.device}")

    if masks is None:
        print("masks: (missing) -> treating all steps as valid")
        valid_mask_flat = torch.ones(n_ep * ep_len, dtype=torch.bool)
    else:
        if not torch.is_tensor(masks):
            raise TypeError(f"'masks' must be a torch.Tensor, got {type(masks)}")
        if masks.shape[:2] != states.shape[:2]:
            raise ValueError(f"masks shape {tuple(masks.shape)} does not match states shape[:2] {tuple(states.shape[:2])}")
        valid_mask_flat = (masks.reshape(-1).float() > 0.5)
        print(f"masks: shape={tuple(masks.shape)} dtype={masks.dtype} valid_steps={int(valid_mask_flat.sum().item())}/{valid_mask_flat.numel()}")

    # Flatten to (N*T, D)
    states_flat = states.reshape(-1, d).to(torch.float32)
    valid_indices = valid_mask_flat.nonzero(as_tuple=False).squeeze(1)
    valid_states = states_flat[valid_mask_flat]
    print(f"valid_states: shape={tuple(valid_states.shape)} dtype={valid_states.dtype}")

    # Basic sanity checks
    any_nan = torch.isnan(valid_states).any().item()
    any_inf = torch.isinf(valid_states).any().item()
    print(f"Contains NaN: {any_nan} | Contains Inf: {any_inf}")

    # Global outliers across ALL dims
    topk = max(0, int(args.topk))
    if topk > 0:
        abs_all = valid_states.abs().reshape(-1)
        k = min(topk, abs_all.numel())
        top_vals, top_idx = torch.topk(abs_all, k=k, largest=True, sorted=True)
        row_idx = top_idx // d
        dim_idx = top_idx % d
        orig_flat_idx = valid_indices[row_idx]
        ep_idx, t_idx = _idx_to_episode_step(orig_flat_idx, ep_len)
        print("\n=== Global top-|x| values (across all dims) ===")
        for i in range(k):
            v = valid_states[row_idx[i], dim_idx[i]].item()
            av = top_vals[i].item()
            print(
                f"#{i+1:02d} | abs={_format_float(av)} val={_format_float(v)} | "
                f"ep={int(ep_idx[i])} t={int(t_idx[i])} dim={int(dim_idx[i])}"
            )

    # Per-dimension summary
    q_list = _parse_floats(args.quantiles)
    q = torch.tensor(q_list, dtype=torch.float32)

    # Compute mean/std/min/max per dim on finite only
    finite_mask = torch.isfinite(valid_states)
    finite_counts = finite_mask.sum(dim=0).to(torch.int64)
    nan_counts = torch.isnan(valid_states).sum(dim=0).to(torch.int64)
    inf_counts = torch.isinf(valid_states).sum(dim=0).to(torch.int64)

    # Replace non-finite with 0 for sums; track counts.
    safe = torch.where(finite_mask, valid_states, torch.zeros_like(valid_states))
    means = safe.sum(dim=0) / torch.clamp(finite_counts.to(torch.float32), min=1.0)
    # variance = E[x^2] - (E[x])^2
    ex2 = (safe * safe).sum(dim=0) / torch.clamp(finite_counts.to(torch.float32), min=1.0)
    vars_ = ex2 - means * means
    vars_ = torch.clamp(vars_, min=0.0)
    stds = torch.sqrt(vars_)

    # min/max: compute on finite values only by masking with +/-inf
    pos_inf = torch.tensor(float("inf"))
    neg_inf = torch.tensor(float("-inf"))
    masked_for_min = torch.where(finite_mask, valid_states, pos_inf)
    masked_for_max = torch.where(finite_mask, valid_states, neg_inf)
    mins = masked_for_min.min(dim=0).values
    maxs = masked_for_max.max(dim=0).values

    # Quantiles per dim (can be slow but D is small)
    # We compute each dim separately so we only sort finite values.
    quantiles_per_dim = torch.empty((q.numel(), d), dtype=torch.float32)
    for dim in range(d):
        col = valid_states[:, dim]
        col = col[torch.isfinite(col)]
        if col.numel() == 0:
            quantiles_per_dim[:, dim] = float("nan")
        else:
            quantiles_per_dim[:, dim] = torch.quantile(col, q)

    print("\n=== Per-dimension summary (valid steps only) ===")
    header_q = " ".join([f"q{qq:g}" for qq in q_list])
    print(f"dim | mean std min max | nan inf | {header_q}")
    for dim in range(d):
        qvals = " ".join([_format_float(x.item()) for x in quantiles_per_dim[:, dim]])
        print(
            f"{dim:02d} | "
            f"{_format_float(means[dim].item())} "
            f"{_format_float(stds[dim].item())} "
            f"{_format_float(mins[dim].item())} "
            f"{_format_float(maxs[dim].item())} | "
            f"{int(nan_counts[dim])} {int(inf_counts[dim])} | "
            f"{qvals}"
        )

    # Threshold counts per dim
    thresholds = _parse_floats(args.thresholds)
    if thresholds:
        print("\n=== Per-dimension |x| threshold counts (valid steps only) ===")
        for thr in thresholds:
            cnt = (valid_states.abs() > thr).sum(dim=0).to(torch.int64)
            total = valid_states.shape[0]
            # print compact: dims where count > 0
            dims = [(i, int(cnt[i])) for i in range(d) if int(cnt[i]) > 0]
            dims_str = ", ".join([f"d{i}:{c}" for i, c in dims[:40]])
            more = "" if len(dims) <= 40 else f" ... (+{len(dims)-40} dims)"
            print(f"|x| > {thr:g}: {len(dims)}/{d} dims have outliers (total_rows={total}); {dims_str}{more}")

    # Sentinel counts per dim
    sentinels = _parse_floats(args.sentinels)
    if sentinels:
        print("\n=== Per-dimension sentinel exact-match counts (valid steps only) ===")
        for s in sentinels:
            s_tensor = torch.tensor(s, dtype=valid_states.dtype)
            cnt = (valid_states == s_tensor).sum(dim=0).to(torch.int64)
            dims = [(i, int(cnt[i])) for i in range(d) if int(cnt[i]) > 0]
            dims_str = ", ".join([f"d{i}:{c}" for i, c in dims[:40]])
            more = "" if len(dims) <= 40 else f" ... (+{len(dims)-40} dims)"
            total = int(cnt.sum().item())
            print(f"value == {s:g}: total_matches={total}; {dims_str}{more}")

    # Step-wise max-|x| diagnostics (helps locate time-local outliers)
    if masks is not None:
        max_abs_per_t = []
        for t in range(ep_len):
            m_t = (masks[:, t].float() > 0.5)
            if int(m_t.sum().item()) == 0:
                max_abs_per_t.append(float("nan"))
                continue
            st_t = states[:, t, :].to(torch.float32)[m_t]
            max_abs_per_t.append(float(st_t.abs().max().item()))
        max_abs_t = torch.tensor(max_abs_per_t)
        top_tk = min(10, ep_len)
        # ignore NaNs by replacing with -inf
        max_abs_t_safe = torch.where(torch.isnan(max_abs_t), torch.tensor(float("-inf")), max_abs_t)
        top_vals_t, top_idx_t = torch.topk(max_abs_t_safe, k=top_tk, largest=True, sorted=True)
        print("\n=== Step-wise max |x| (top steps) ===")
        for i in range(top_tk):
            t = int(top_idx_t[i].item())
            v = float(top_vals_t[i].item())
            if math.isinf(v) and v < 0:
                continue
            print(f"t={t:03d} max_abs={_format_float(v)}")


if __name__ == "__main__":
    main()


