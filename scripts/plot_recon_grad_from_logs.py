#!/usr/bin/env python3
"""
Parse training logs and plot recon loss + grad curves.

Expected log lines (examples):
  Epoch 9746 | Update 9747 | ELBO: -1.2511 | ReconNLL: -8.6350 | KL: 0.2118 | ActionFM: 7.2528 | Grad: 87.2609 | Valid steps: 23040.0
  Epoch 9750 | Update 9751 | ELBO: -18.5849 | ReconNLL: -20.1232 | KL: 1.5383 | Grad: 49.4868 | Valid steps: 23040.0

ReconNLL sign can differ across runs/logging versions (sometimes positive NLL, sometimes
negative log-likelihood). We therefore plot recon loss as abs(ReconNLL) so it's always
positive and "lower is better".
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import matplotlib. Please ensure it's installed in your Python env."
    ) from e


LINE_RE = re.compile(
    r"Update\s+(?P<update>\d+)\s+\|.*?ReconNLL:\s+(?P<recon>[-+]?\d+(?:\.\d+)?)"
    r".*?Grad:\s+(?P<grad>[-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class Series:
    update: List[int]
    recon_loss: List[float]
    grad: List[float]


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out: List[float] = []
    running = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def parse_log(path: str) -> Series:
    updates: List[int] = []
    recon_losses: List[float] = []
    grads: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            upd = int(m.group("update"))
            recon_nll = float(m.group("recon"))
            grad = float(m.group("grad"))
            updates.append(upd)
            recon_losses.append(abs(recon_nll))  # force positive, comparable "loss"
            grads.append(grad)

    # Sort by update in case log ordering is odd
    pairs = sorted(zip(updates, recon_losses, grads), key=lambda x: x[0])
    if not pairs:
        raise ValueError(f"No matching lines found in log: {path}")

    su, sr, sg = zip(*pairs)
    return Series(update=list(su), recon_loss=list(sr), grad=list(sg))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, series: Series) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["update", "recon_loss", "grad"])
        w.writerows(zip(series.update, series.recon_loss, series.grad))


def plot_two(
    *,
    x1: List[int],
    y1: List[float],
    label1: str,
    x2: List[int],
    y2: List[float],
    label2: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label=label1, linewidth=1.8)
    plt.plot(x2, y2, label=label2, linewidth=1.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory for plots/CSVs")
    ap.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Moving average window (in updates). Use 1 to disable smoothing.",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    # Tasks (envs) to compare: baseline (VAE) vs CVAE
    # Keep this list minimal so the script outputs exactly what we want.
    tasks: List[Tuple[str, str, str]] = [
        (
            "ShellGameTouch-v0",
            "/local/s4176650/sphinx/storage/ShellGameTouch-v0_vae_mikasa_last5/log.txt",
            "/local/s4176650/sphinx/storage_cvae/ShellGameTouch-v0_cvae_mikasa_last5_v6_1/log.txt",
        ),
        (
            "ShellGamePush-v0",
            "/local/s4176650/sphinx/storage/ShellGamePush-v0_vae_mikasa_last5/log.txt",
            "/local/s4176650/sphinx/storage_cvae/ShellGamePush-v0_cvae_mikasa_last5_v6_1/log.txt",
        ),
    ]

    for env_name, baseline_path, cvae_path in tasks:
        baseline = parse_log(baseline_path)
        cvae = parse_log(cvae_path)

        # Export parsed series for transparency/debugging
        write_csv(os.path.join(out_dir, f"{env_name}__baseline_vae.csv"), baseline)
        write_csv(os.path.join(out_dir, f"{env_name}__cvae.csv"), cvae)

        b_recon = moving_average(baseline.recon_loss, args.smooth)
        c_recon = moving_average(cvae.recon_loss, args.smooth)
        b_grad = moving_average(baseline.grad, args.smooth)
        c_grad = moving_average(cvae.grad, args.smooth)

        plot_two(
            x1=baseline.update,
            y1=b_recon,
            label1="baseline (VAE)",
            x2=cvae.update,
            y2=c_recon,
            label2="CVAE",
            title=f"{env_name} - Recon loss (-ReconNLL), smooth={args.smooth}",
            xlabel="Update",
            ylabel="Recon loss",
            out_path=os.path.join(out_dir, f"{env_name}__recon_loss.png"),
        )

        plot_two(
            x1=baseline.update,
            y1=b_grad,
            label1="baseline (VAE)",
            x2=cvae.update,
            y2=c_grad,
            label2="CVAE",
            title=f"{env_name} - Grad, smooth={args.smooth}",
            xlabel="Update",
            ylabel="Grad",
            out_path=os.path.join(out_dir, f"{env_name}__grad.png"),
        )

    print(f"Done. Wrote 4 plots + 4 CSVs to: {out_dir}")


if __name__ == "__main__":
    main()

