import argparse
import os
import random
import torch


EP_KEYS_CANDIDATES = ["episodes", "trajectories", "rollouts", "demos", "data"]
STEP_DICT_KEYS_CANDIDATES = ["obss", "states", "joints", "actions", "rewards", "masks"]


def _is_tensor_like(x):
    return torch.is_tensor(x)


def _leading_len(x):
    """
    Best-effort "time dimension length" for tensor-like/list-like objects.
    Returns int or None if unknown.
    """
    if torch.is_tensor(x):
        return int(x.shape[0])
    if isinstance(x, (list, tuple)):
        return len(x)
    return None


def _flatten_masks(masks):
    if torch.is_tensor(masks):
        m = masks.detach().cpu()
        # keep dims; we'll interpret masks more carefully later
        return m
    raise TypeError(f"Unsupported masks type: {type(masks)}")


def _flatten_1d_signal(x, name="signal"):
    """
    Convert a tensor into a 1D (T,) tensor by squeezing trivial dims.
    Raises if it cannot be interpreted as a per-time-step signal.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported {name} type: {type(x)}")
    t = x.detach().cpu()
    # squeeze trailing singleton dims (e.g., [T,1] or [T,1,1])
    while t.ndim >= 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    # accept [T] or [T,1]
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    if t.ndim != 1:
        raise TypeError(f"Cannot flatten {name} to 1D; got shape={tuple(t.shape)}")
    return t


def _episode_slices_from_end_idxs(T, end_idxs_inclusive):
    """
    Build episode slices where each end index is inclusive.
    Returns list of (start, end_exclusive).
    """
    end_idxs = [int(e) for e in end_idxs_inclusive if 0 <= int(e) < int(T)]
    end_idxs = sorted(set(end_idxs))
    slices = []
    start = 0
    for end in end_idxs:
        end_excl = end + 1
        if end_excl > start:
            slices.append((start, end_excl))
        start = end_excl
    if start < int(T):
        slices.append((start, int(T)))
    return slices


def _episode_slices_from_start_idxs(T, start_idxs):
    """
    Build episode slices where each start index indicates the FIRST step of an episode.
    Returns list of (start, end_exclusive).
    """
    starts = [int(s) for s in start_idxs if 0 <= int(s) < int(T)]
    starts = sorted(set(starts))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    slices = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else int(T)
        if e > s:
            slices.append((s, e))
    return slices


def _episode_slices_from_masks(masks):
    """
    Convert a vector/tensor of masks into episode slices.

    Different codebases encode masks differently:
    - sometimes masks[t] == 0 at terminal steps (episode end)
    - sometimes masks[t] == 0 at reset steps (episode start)

    We try BOTH interpretations and let the caller decide which one is more plausible.
    Returns list of (start, end_exclusive).
    """
    m = _flatten_masks(masks)
    if m.ndim == 1 or (m.ndim == 2 and m.shape[1] == 1):
        m1 = _flatten_1d_signal(m, name="masks")
        T = int(m1.shape[0])
        zeros = (m1 == 0).nonzero(as_tuple=False).view(-1).tolist()
        # candidate A: zeros are episode ends
        end_slices = _episode_slices_from_end_idxs(T, zeros)
        # candidate B: zeros are episode starts
        start_slices = _episode_slices_from_start_idxs(T, zeros)
        return {
            "mask_zero_is_end": end_slices,
            "mask_zero_is_start": start_slices,
            "T": T,
            "zeros": len(zeros),
        }

    # If masks is 2D [T, N], we don't guess how to split per-env here.
    # Fall back to reward-based or fixed-length splitting later.
    return {"mask_unsupported_ndim": True, "shape": tuple(m.shape)}


def _episode_slices_from_rewards(rewards):
    """
    Heuristic: many tasks emit a non-zero reward at episode termination.
    We treat any step where reward != 0 as episode end (inclusive).
    """
    r1 = _flatten_1d_signal(rewards, name="rewards")
    T = int(r1.shape[0])
    end_idxs = (r1 != 0).nonzero(as_tuple=False).view(-1).tolist()
    slices = _episode_slices_from_end_idxs(T, end_idxs)
    return {"reward_nonzero_is_end": slices, "T": T, "nonzero": len(end_idxs)}


def _episode_slices_fixed_len(T, episode_len):
    L = int(episode_len)
    if L <= 0:
        raise ValueError(f"episode_len must be > 0, got {episode_len}")
    slices = [(s, min(s + L, int(T))) for s in range(0, int(T), L)]
    # drop last if it's too short? keep it to avoid losing data
    return slices


def _batched_episode_count_from_step_dict(obj):
    """
    Detect dict-of-tensors shaped like (N_episodes, T, ...) for step keys.
    Returns N if consistent, else None.
    """
    if not isinstance(obj, dict):
        return None
    # need at least a couple keys to be meaningful
    present = [k for k in STEP_DICT_KEYS_CANDIDATES if k in obj and torch.is_tensor(obj[k])]
    if len(present) < 2:
        return None

    Ns = []
    for k in present:
        v = obj[k]
        if not torch.is_tensor(v) or v.ndim < 2:
            return None
        Ns.append(int(v.shape[0]))
    if len(set(Ns)) != 1:
        return None
    return Ns[0]


def _index_select0(value, idx_tensor):
    """
    Select by episode index along dim0 for tensors / nested dicts; keep metadata unchanged.
    """
    if torch.is_tensor(value):
        return value.index_select(0, idx_tensor)
    if isinstance(value, dict):
        return {k: _index_select0(v, idx_tensor) for k, v in value.items()}
    # Non-tensor metadata (strings, scalars, etc.)
    return value


def _select_by_slices(value, slices):
    """
    Select and concatenate chunks from `value` along dim0, using episode slices.
    Supports: torch.Tensor, dict (recursive), list/tuple (if length matches time dim).
    """
    if torch.is_tensor(value):
        chunks = [value[s:e] for (s, e) in slices]
        return torch.cat(chunks, dim=0) if len(chunks) else value[:0]

    if isinstance(value, dict):
        return {k: _select_by_slices(v, slices) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        chunks = []
        for (s, e) in slices:
            chunks.extend(value[s:e])
        return chunks

    # Non-indexable metadata (e.g., scalars/strings): keep as-is
    return value


def extract_episodes(obj, *, min_episodes=None, episode_len=None):
    """
    Returns (episodes, meta)

    - episodes: list-like "episode units" you can sample (either actual episodes OR (start,end) slices)
    - meta: dict describing how to rebuild/save
    """
    # Format A: already a list of episodes
    if isinstance(obj, (list, tuple)):
        return list(obj), {"kind": "episode_list", "container": obj, "ep_key": None}

    if isinstance(obj, dict):
        # Format B: dict with explicit episodes list under a known key
        for k in EP_KEYS_CANDIDATES:
            if k in obj and isinstance(obj[k], (list, tuple)):
                return list(obj[k]), {"kind": "dict_episode_list", "container": obj, "ep_key": k}

        # Format C0: batched-by-episode dict of tensors: (N, T, ...)
        N = _batched_episode_count_from_step_dict(obj)
        if N is not None:
            return list(range(N)), {"kind": "batched_step_dict", "container": obj, "N": N}

        # Format C: per-step dict (e.g. obss/actions/rewards/masks) + masks define episode boundaries
        if "masks" in obj and any(k in obj for k in STEP_DICT_KEYS_CANDIDATES):
            candidates = []
            # 1) masks-based candidates (try both semantics)
            try:
                mask_info = _episode_slices_from_masks(obj["masks"])
                if "mask_zero_is_end" in mask_info:
                    candidates.append(("masks_zero_is_end", mask_info["mask_zero_is_end"]))
                    candidates.append(("masks_zero_is_start", mask_info["mask_zero_is_start"]))
            except Exception:
                pass

            # 2) rewards-based candidate (often reliable if masks are all-ones)
            try:
                if "rewards" in obj:
                    r_info = _episode_slices_from_rewards(obj["rewards"])
                    candidates.append(("rewards_nonzero_is_end", r_info["reward_nonzero_is_end"]))
            except Exception:
                pass

            # pick best candidate
            if candidates:
                # prefer candidates that meet min_episodes; otherwise pick the one with max episodes
                best_name, best_slices = None, None
                if min_episodes is not None:
                    feasible = [(n, name, slc) for (name, slc) in candidates for n in [len(slc)] if n >= int(min_episodes)]
                    if feasible:
                        feasible.sort(key=lambda x: x[0], reverse=True)
                        _, best_name, best_slices = feasible[0]
                if best_slices is None:
                    candidates_sorted = sorted([(len(slc), name, slc) for (name, slc) in candidates], reverse=True)
                    _, best_name, best_slices = candidates_sorted[0]

                return best_slices, {"kind": "step_dict", "container": obj, "split_strategy": best_name}

            # 3) final fallback: fixed length if provided
            # infer T from any tensor-like entry
            T = None
            for k in STEP_DICT_KEYS_CANDIDATES:
                if k in obj and torch.is_tensor(obj[k]):
                    T = int(obj[k].shape[0])
                    break
            if T is not None and episode_len is not None:
                slices = _episode_slices_fixed_len(T, episode_len)
                return slices, {"kind": "step_dict", "container": obj, "split_strategy": f"fixed_len_{episode_len}"}

            raise TypeError(
                "step-dict dataset detected but could not infer episode boundaries.\n"
                "Try providing --episode_len <H> (fixed horizon per episode), or inspect masks/rewards semantics.\n"
                f"keys={list(obj.keys())}"
            )

    raise TypeError(
        "Unsupported .pt structure for episode sampling.\n"
        f"type={type(obj)}\n"
        f"Got keys={list(obj.keys()) if isinstance(obj, dict) else None}\n"
        "Supported formats:\n"
        f"- list/tuple of episodes\n"
        f"- dict with one of keys {EP_KEYS_CANDIDATES} as list/tuple\n"
        f"- per-step dict with 'masks' and step keys like {STEP_DICT_KEYS_CANDIDATES}\n"
    )


def rebuild_from_selected(selected, meta):
    kind = meta["kind"]
    if kind == "episode_list":
        return selected
    if kind == "dict_episode_list":
        out = dict(meta["container"])
        out[meta["ep_key"]] = selected
        return out
    if kind == "batched_step_dict":
        # selected is list of episode indices
        idx = torch.tensor(selected, dtype=torch.long)
        src = meta["container"]
        return {k: _index_select0(v, idx) for k, v in src.items()}
    if kind == "step_dict":
        src = meta["container"]
        # selected is list of (start,end) slices
        out = {}
        for k, v in src.items():
            out[k] = _select_by_slices(v, selected)
        return out
    raise ValueError(f"Unknown meta kind: {kind}")


def build_mixed_output(mixed_units, meta_expert, meta_random):
    """
    Build final output object from a mixed list of episode-units.
    - For episode_list/dict_episode_list: mixed_units are the episodes themselves.
    - For step_dict: mixed_units are tuples: (source_container_dict, (start,end))
    """
    if meta_expert["kind"] != meta_random["kind"]:
        raise ValueError(f"Cannot mix different dataset kinds: {meta_expert['kind']} vs {meta_random['kind']}")

    kind = meta_random["kind"]
    if kind in ("episode_list", "dict_episode_list"):
        # mixed_units are already episodes; keep container format of random side
        return rebuild_from_selected(mixed_units, meta_random)

    if kind == "batched_step_dict":
        # mixed_units are tuples: (source_container_dict, episode_idx)
        src_random = meta_random["container"]
        # Build an index list per-source, then select & concat.
        # (This avoids per-episode Python slicing; uses vectorized index_select)
        expert_src = meta_expert["container"]
        random_src = meta_random["container"]
        expert_idx = [i for (src, i) in mixed_units if src is expert_src]
        random_idx = [i for (src, i) in mixed_units if src is random_src]

        out = {}
        for k in src_random.keys():
            ve = _index_select0(expert_src[k], torch.tensor(expert_idx, dtype=torch.long)) if expert_idx else None
            vr = _index_select0(random_src[k], torch.tensor(random_idx, dtype=torch.long)) if random_idx else None

            if ve is None:
                out[k] = vr
            elif vr is None:
                out[k] = ve
            else:
                if torch.is_tensor(ve) and torch.is_tensor(vr):
                    out[k] = torch.cat([ve, vr], dim=0)
                elif isinstance(ve, dict) and isinstance(vr, dict):
                    out[k] = {kk: torch.cat([ve[kk], vr[kk]], dim=0) for kk in vr.keys()}
                else:
                    # metadata: prefer random
                    out[k] = vr

        return out

    if kind == "step_dict":
        # mixed_units: [(src_dict, (s,e)), ...]
        src_random = meta_random["container"]
        out = {}
        for k in src_random.keys():
            # gather slices per unit from their own source dict
            pieces = []
            for src_dict, (s, e) in mixed_units:
                v = src_dict[k]
                if torch.is_tensor(v):
                    pieces.append(v[s:e])
                elif isinstance(v, dict):
                    # nested dict (e.g., obss may be dict of tensors)
                    pieces.append(_select_by_slices(v, [(s, e)]))
                elif isinstance(v, (list, tuple)):
                    pieces.append(list(v[s:e]))
                else:
                    # metadata: just use random's version (first time)
                    pieces = [src_random[k]]
                    break

            if not pieces:
                out[k] = src_random[k]
            else:
                if torch.is_tensor(pieces[0]):
                    out[k] = torch.cat(pieces, dim=0)
                elif isinstance(pieces[0], dict):
                    # _select_by_slices returns dict; concatenate per leaf by reusing _select_by_slices with each slice
                    # Here we already have dict-per-piece, so merge by concatenating matching keys.
                    merged = {}
                    for subk in pieces[0].keys():
                        subv = [p[subk] for p in pieces]
                        merged[subk] = torch.cat(subv, dim=0) if torch.is_tensor(subv[0]) else subv
                    out[k] = merged
                elif isinstance(pieces[0], list):
                    merged_list = []
                    for p in pieces:
                        merged_list.extend(p)
                    out[k] = merged_list
                else:
                    out[k] = pieces[0]

        return out

    raise ValueError(f"Unknown meta kind: {kind}")


def sample_without_replacement(items, k, rng):
    if k > len(items):
        raise ValueError(f"Requested k={k}, but only have n={len(items)} episodes.")
    idx = rng.sample(range(len(items)), k)
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_expert", type=int, default=1000)
    ap.add_argument("--n_random", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_out", action="store_true", help="Shuffle merged episodes before saving.")
    ap.add_argument("--episode_len", type=int, default=None, help="Fallback: fixed episode length (used only if boundaries can't be inferred).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    expert_obj = torch.load(args.expert, map_location="cpu")
    random_obj = torch.load(args.random, map_location="cpu")

    expert_units, expert_meta = extract_episodes(expert_obj, min_episodes=args.n_expert, episode_len=args.episode_len)
    random_units, random_meta = extract_episodes(random_obj, min_episodes=args.n_random, episode_len=args.episode_len)

    expert_sel = sample_without_replacement(expert_units, args.n_expert, rng)
    random_sel = sample_without_replacement(random_units, args.n_random, rng)

    if random_meta["kind"] != expert_meta["kind"]:
        raise ValueError(f"expert kind={expert_meta['kind']} != random kind={random_meta['kind']}, cannot mix safely.")

    if random_meta["kind"] in ("episode_list", "dict_episode_list"):
        mixed_units = expert_sel + random_sel
        if args.shuffle_out:
            rng.shuffle(mixed_units)
        out_obj = build_mixed_output(mixed_units, expert_meta, random_meta)
        mixed_count = len(mixed_units)

    elif random_meta["kind"] == "batched_step_dict":
        expert_src = expert_meta["container"]
        random_src = random_meta["container"]

        expert_pairs = [(expert_src, i) for i in expert_sel]
        random_pairs = [(random_src, i) for i in random_sel]
        mixed_pairs = expert_pairs + random_pairs

        if args.shuffle_out:
            rng.shuffle(mixed_pairs)

        out_obj = build_mixed_output(mixed_pairs, expert_meta, random_meta)
        mixed_count = len(mixed_pairs)

        # If we shuffled, the concat in build_mixed_output groups by source; apply a final permutation to match mixed order.
        # This keeps semantics consistent with --shuffle_out while remaining efficient.
        if args.shuffle_out:
            # build a stable permutation by mapping (src,id) order to concat order
            # concat order is [expert_sel..., random_sel...] as they appear in mixed_pairs filtering
            # easiest: rebuild by selecting from concatenated output using the mixed order indices.
            # We'll do it with a single index_select on dim0.
            # Determine position of each pair in the concat result:
            exp_map = {i: pos for pos, i in enumerate([i for (src, i) in mixed_pairs if src is expert_src])}
            rnd_base = len(exp_map)
            rnd_map = {i: rnd_base + pos for pos, i in enumerate([i for (src, i) in mixed_pairs if src is random_src])}
            positions = []
            exp_seen = {}
            rnd_seen = {}
            # handle potential duplicates defensively
            for src, i in mixed_pairs:
                if src is expert_src:
                    c = exp_seen.get(i, 0)
                    exp_seen[i] = c + 1
                    if c > 0:
                        raise ValueError("Duplicate expert episode index in mixed list; this should not happen.")
                    positions.append(exp_map[i])
                else:
                    c = rnd_seen.get(i, 0)
                    rnd_seen[i] = c + 1
                    if c > 0:
                        raise ValueError("Duplicate random episode index in mixed list; this should not happen.")
                    positions.append(rnd_map[i])

            perm = torch.tensor(positions, dtype=torch.long)
            out_obj = {k: _index_select0(v, perm) for k, v in out_obj.items()}

    elif random_meta["kind"] == "step_dict":
        # represent each selected episode as (source_container_dict, (start,end))
        expert_chunks = [(expert_meta["container"], slc) for slc in expert_sel]
        random_chunks = [(random_meta["container"], slc) for slc in random_sel]
        mixed_chunks = expert_chunks + random_chunks
        if args.shuffle_out:
            rng.shuffle(mixed_chunks)
        out_obj = build_mixed_output(mixed_chunks, expert_meta, random_meta)
        mixed_count = len(mixed_chunks)

    else:
        raise ValueError(f"Unknown kind: {random_meta['kind']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_obj, args.out)

    print("Saved:", args.out)
    if expert_meta.get("split_strategy"):
        print("expert split_strategy:", expert_meta["split_strategy"])
    if random_meta.get("split_strategy"):
        print("random split_strategy:", random_meta["split_strategy"])
    print("expert kind:", expert_meta["kind"], "episodes:", len(expert_units), "sampled:", len(expert_sel))
    print("random kind:", random_meta["kind"], "episodes:", len(random_units), "sampled:", len(random_sel))
    print("mixed episodes:", mixed_count)


if __name__ == "__main__":
    main()