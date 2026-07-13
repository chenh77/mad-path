#!/usr/bin/env python3
"""
Violin plot + summary table comparing MAD-path and APT+DEO on the
Bayesian GMM (Old Faithful) target across multiple seeds (paper Figure 6).

Plot: mode-1 weight  (true = 0.5, dashed line).
Summary table additionally reports (swap) acceptance rate and run time.

Usage:
    python plot_bayesian_gmm.py \
        --results_dir experiments/results/bayesian_gmm/ \
        --output      experiments/plots/bayesian_gmm.pdf
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.color":        "#cccccc",
    "grid.alpha":        0.7,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

METHODS = [
    ("MAD-path", "mad",     "#999999"),
    ("APT-DEO",  "apt_deo", "#999999"),
]

METRICS = [
    ("mode1_weight",     "Mode-1 weight",        0.5),
    ("time_s",           "Wall-clock time (s)",   None),
    ("swap_rate",        "Acceptance rate",        None),
]

# Keys that don't follow the {prefix}_{metric} pattern
_KEY_ALIASES = {
    "mad_swap_rate": "acc_mad",
}


def load_results(results_dir: str) -> dict[str, list[float]]:
    files = sorted(glob.glob(os.path.join(results_dir, "seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"No seed_*.npz files found in {results_dir}")
    print(f"Loading {len(files)} result files...")

    data: dict[str, list[float]] = {
        f"{prefix}_{metric}": [] for _, prefix, _ in METHODS
        for metric, _, _ in METRICS
    }

    for path in files:
        d = np.load(path)
        for _, prefix, _ in METHODS:
            for metric, _, _ in METRICS:
                key     = f"{prefix}_{metric}"
                src_key = _KEY_ALIASES.get(key, key)
                if src_key in d:
                    data[key].append(float(d[src_key]))
                else:
                    print(f"  Warning: key '{key}' missing in {os.path.basename(path)}")

    return data


def plot(data: dict[str, list[float]], output: str) -> None:
    metrics_to_plot = METRICS[:1]  # only mode-1-weight
    fig, axes = plt.subplots(1, 1, figsize=(3.0, 2.4))
    axes = np.atleast_1d(axes)

    for ax, (metric, ylabel, true_val) in zip(axes, metrics_to_plot):
        # Gather per-method arrays in display order
        values = [np.array(data[f"{prefix}_{metric}"]) for _, prefix, _ in METHODS]
        labels = [name for name, _, _ in METHODS]
        colors = [color for _, _, color in METHODS]

        positions = range(1, len(METHODS) + 1)
        vp = ax.violinplot(values, positions=positions, widths=0.5,
                           showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)


        if true_val is not None:
            ax.axhline(true_val, color="#C44E52", linewidth=1.2,
                       linestyle="--", zorder=3, label=f"true = {true_val}")
            # ax.legend(fontsize=7.5, framealpha=0.8)

        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # # Annotate sample sizes
        # for i, vals in enumerate(values, start=1):
        #     ax.text(i, ax.get_ylim()[0], f"n={len(vals)}",
        #             ha="center", va="bottom", fontsize=6.5, color="#555555")

    # fig.suptitle("Bayesian GMM robustness: MAD-path vs APT variants", fontsize=10, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.show()


def print_summary(data: dict[str, list[float]]) -> None:
    print(f"\n{'Method':<12}  {'Metric':<30}  {'median':>8}  {'mean':>8}  {'std':>8}  n")
    print("-" * 75)
    for metric, ylabel, _ in METRICS:
        for name, prefix, _ in METHODS:
            vals = np.array(data[f"{prefix}_{metric}"])
            if len(vals):
                print(f"{name:<12}  {ylabel:<30}  "
                      f"{np.median(vals):8.4f}  {np.mean(vals):8.4f}  "
                      f"{np.std(vals):8.4f}  {len(vals)}")
        print()


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", default=os.path.join(_SCRIPT_DIR, "results", "bayesian_gmm"),
    )
    parser.add_argument(
        "--output", default=os.path.join(_SCRIPT_DIR, "plots", "bayesian_gmm.pdf"),
    )
    args = parser.parse_args()

    data = load_results(args.results_dir)
    print_summary(data)
    plot(data, args.output)
