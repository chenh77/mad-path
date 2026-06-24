#!/usr/bin/env python3
"""
Plot 2D marginals and 1D trace plots for SUR Case 2 results.

Loads the .npz produced by run_sur.py and generates two figures:

Figure 2 — 1D marginals (3 rows × 5 cols):
  Row order: MAD-Path, APT-DEO (warm 1), APT-DEO (warm 2)
  NUTS shown as dashed reference overlay on every panel.
  Samples after burn-in only.

Figure 3 — Early trace plots:
  First 500 samples of MAD-Path, APT-DEO (warm 1), APT-DEO (cold) overlaid
  per dimension, including burn-in phase to show phase transition.

Usage (from the repo root):
    python experiments/plot_sur.py
    python experiments/plot_sur.py \\
        --results experiments/results/sur/case2.npz \\
        --output_dir experiments/plots/sur/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid":         True,
    "grid.linewidth":    0.35,
    "grid.color":        "#cccccc",
    "grid.alpha":        0.7,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# Must match BURNIN_APT in run_sur.py (used to slice post-burnin for figure 2)
BURNIN_APT = 2000

METHOD_COLORS = {
    "MAD-Path":          "darkgreen",
    "APT-DEO (warm 1)":  "crimson",
    "APT-DEO (warm 2)":  "darkorange",
    "APT-DEO (cold)":    "steelblue",
    "NUTS":              "black",
}
MODE_COLOR = "#C44E52"


def make_figure_1d_marginals(
    chain_mad: np.ndarray,
    chain_apt_warm1: np.ndarray,
    chain_apt_warm2: np.ndarray,
    modes: np.ndarray,
    param_names: list[str],
    chain_nuts: np.ndarray | None = None,
) -> plt.Figure:
    """Row order: MAD-Path, APT-DEO (warm 1), APT-DEO (warm 2).
    NUTS shown as dashed reference overlay. All chains sliced to post-burnin."""
    from scipy.stats import gaussian_kde

    D = chain_mad.shape[1]
    methods = [
        ("MAD-Path",           chain_mad,        METHOD_COLORS["MAD-Path"]),
        ("APT-DEO 1",  chain_apt_warm1,  METHOD_COLORS["APT-DEO (warm 1)"]),
        ("APT-DEO 2",  chain_apt_warm2,  METHOD_COLORS["APT-DEO (warm 2)"]),
    ]
    n_rows = len(methods)

    # Pre-compute NUTS KDE per dimension (used as reference overlay)
    nuts_kdes = []
    if chain_nuts is not None:
        for di in range(D):
            nuts_kdes.append(gaussian_kde(chain_nuts[:, di], bw_method="scott"))

    fig, axes = plt.subplots(n_rows, D, figsize=(2.2 * D, 1.6 * n_rows), sharey=False)
    axes = np.array(axes)

    for row, (method_name, chain, color) in enumerate(methods):
        for di in range(D):
            ax = axes[row, di]
            ax.set_axisbelow(True)
            samples = chain[:, di]
            kde = gaussian_kde(samples, bw_method="scott")
            xs = np.linspace(samples.min(), samples.max(), 400)
            ax.plot(xs, kde(xs), lw=1.2, color=color)
            ax.fill_between(xs, kde(xs), alpha=0.18, color=color)
            if nuts_kdes:
                nuts_xs = np.linspace(chain_nuts[:, di].min(), chain_nuts[:, di].max(), 400)
                ax.plot(nuts_xs, nuts_kdes[di](nuts_xs),
                        lw=1.0, color=METHOD_COLORS["NUTS"], ls="--", alpha=0.55,
                        label="NUTS" if (row == 0 and di == 0) else None)
            # for m in modes:
            #     ax.axvline(m[di], color=MODE_COLOR, lw=0.9, ls="--", alpha=0.8)
            ax.set_xlabel(param_names[di], fontsize=8)
            ax.tick_params(labelsize=7)
            if di == 0:
                ax.set_ylabel(method_name, fontsize=9, labelpad=4)

    # if nuts_kdes:
    #     axes[0, 0].legend(fontsize=8, framealpha=0.8, loc="upper right")

    for di in range(D):
        xmin = min(axes[r, di].get_xlim()[0] for r in range(n_rows))
        xmax = max(axes[r, di].get_xlim()[1] for r in range(n_rows))
        if chain_nuts is not None:
            xmin = min(xmin, chain_nuts[:, di].min())
            xmax = max(xmax, chain_nuts[:, di].max())
        ymin = min(axes[r, di].get_ylim()[0] for r in range(n_rows))
        ymax = max(axes[r, di].get_ylim()[1] for r in range(n_rows))
        for r in range(n_rows):
            axes[r, di].set_xlim(xmin, xmax)
            axes[r, di].set_ylim(ymin, ymax)

    fig.tight_layout()
    return fig


def make_figure_early_traces(
    chain_mad: np.ndarray,
    chain_apt_warm1: np.ndarray,
    chain_apt_cold: np.ndarray,
    modes: np.ndarray,
    param_names: list[str],
    n_show: int = 500,
) -> plt.Figure:
    """Overlay first n_show samples (including burn-in) for MAD-Path,
    APT-DEO (warm 1), APT-DEO (cold) — one subplot per dimension."""
    D = chain_mad.shape[1]

    methods = [
        ("MAD-Path",         chain_mad[:n_show],        METHOD_COLORS["MAD-Path"]),
        ("APT-DEO 1", chain_apt_warm1[:n_show],  METHOD_COLORS["APT-DEO (warm 1)"]),
        ("APT-DEO (cold)",   chain_apt_cold[:n_show],   METHOD_COLORS["APT-DEO (cold)"]),
    ]

    fig, axes = plt.subplots(1, D, figsize=(2.2 * D, 2.0), sharey=False)
    axes = np.array(axes)

    for di in range(D):
        ax = axes[di]
        ax.set_axisbelow(True)
        for method_name, chain, color in methods:
            n = len(chain)
            ax.plot(np.arange(n), chain[:, di],
                    lw=0.6, alpha=0.85, color=color,
                    label=method_name if di == 0 else None,
                    rasterized=True)
        for m in modes:
            ax.axhline(m[di], color=MODE_COLOR, lw=0.8, ls="--", alpha=0.6)
        ax.set_title(param_names[di], fontsize=9)
        ax.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        fontsize=10, framealpha=0,
        bbox_to_anchor=(0.5, -0.05),
    )
    for lh in leg.get_lines():
        lh.set_linewidth(2.5)

    fig.supxlabel("iteration", fontsize=8, y=0.12)
    fig.tight_layout()
    return fig


def main(results_path: str, output_dir: str, mad_burnin: int, trace_n: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    data        = np.load(results_path, allow_pickle=True)
    chain_mad   = data["chain_mad"]
    chain_apt   = data["chain_apt"]    # cold start, full chain
    chain_apt2  = data["chain_apt2"]   # warm start 1, full chain
    chain_apt3  = data["chain_apt3"]   # warm start 2, full chain
    modes       = data["modes"]
    param_names = list(data["param_names"])
    acc_mad     = float(data["acc_mad"])
    swap_rate   = float(data["apt_swap_rate"])
    swap_rate2  = float(data["apt2_swap_rate"])
    swap_rate3  = float(data["apt3_swap_rate"])
    chain_nuts  = data["chain_nuts"] if "chain_nuts" in data else None
    nuts_acc    = float(data["nuts_acc"]) if "nuts_acc" in data else None

    print(f"chain_mad  : {chain_mad.shape}  (accept={acc_mad:.3f})")
    print(f"chain_apt  : {chain_apt.shape}  (swap={swap_rate:.3f})  [cold]")
    print(f"chain_apt2 : {chain_apt2.shape}  (swap={swap_rate2:.3f})  [warm 1]")
    print(f"chain_apt3 : {chain_apt3.shape}  (swap={swap_rate3:.3f})  [warm 2]")
    if chain_nuts is not None:
        print(f"chain_nuts : {chain_nuts.shape}  (mean_acc={nuts_acc:.3f})")
    print(f"modes      : {modes.shape}")

    # Guard against NaN
    for name, chain in [("MAD-Path", chain_mad), ("APT cold", chain_apt),
                        ("APT warm 1", chain_apt2), ("APT warm 2", chain_apt3)]:
        n_bad = (~np.isfinite(chain).all(axis=1)).sum()
        if n_bad:
            print(f"  Warning: {n_bad} NaN rows in {name}")
    chain_mad  = chain_mad[np.isfinite(chain_mad).all(axis=1)]
    chain_apt  = chain_apt[np.isfinite(chain_apt).all(axis=1)]
    chain_apt2 = chain_apt2[np.isfinite(chain_apt2).all(axis=1)]
    chain_apt3 = chain_apt3[np.isfinite(chain_apt3).all(axis=1)]
    if chain_nuts is not None:
        chain_nuts = chain_nuts[np.isfinite(chain_nuts).all(axis=1)]

    # ── Figure 2: 1D marginals (post-burnin) ─────────────────────────────────
    # MAD-Path, APT warm 1, APT warm 2; NUTS as reference overlay
    mad_post       = chain_mad[mad_burnin:]
    apt_warm1_post = chain_apt2[BURNIN_APT:]
    apt_warm2_post = chain_apt3[BURNIN_APT:]

    print(f"\nFigure 2 sample counts (post-burnin):")
    print(f"  MAD-Path         : {len(mad_post)}  (burnin={mad_burnin})")
    print(f"  APT-DEO (warm 1) : {len(apt_warm1_post)}  (burnin={BURNIN_APT})")
    print(f"  APT-DEO (warm 2) : {len(apt_warm2_post)}  (burnin={BURNIN_APT})")

    fig2 = make_figure_1d_marginals(
        mad_post, apt_warm1_post, apt_warm2_post,
        modes, param_names, chain_nuts,
    )
    path2 = os.path.join(output_dir, "sur_case2_1d_marginals.pdf")
    fig2.savefig(path2, bbox_inches="tight", dpi=150)
    print(f"Saved → {path2}")

    # ── Figure 3: early traces (includes burn-in) ─────────────────────────────
    # MAD-Path, APT warm 1, APT cold — shows convergence from different inits
    # n_trace = min(trace_n, len(chain_mad), len(chain_apt2), len(chain_apt))
    n_trace = BURNIN_APT  
    print(f"\nFigure 3: showing first {n_trace} samples (includes burn-in phase)")

    fig3 = make_figure_early_traces(
        chain_mad, chain_apt2, chain_apt,
        modes, param_names, n_show=n_trace,
    )
    path3 = os.path.join(output_dir, "sur_case2_early_traces.pdf")
    fig3.savefig(path3, bbox_inches="tight", dpi=150)
    print(f"Saved → {path3}")

    plt.show()


if __name__ == "__main__":
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _default_res = os.path.join(_script_dir, "results", "sur", "case2.npz")
    _default_out = os.path.join(_script_dir, "plots")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    type=str, default=_default_res)
    parser.add_argument("--output_dir", type=str, default=_default_out)
    parser.add_argument("--mad_burnin", type=int, default=0,
                        help="Burn-in to drop from MAD-Path for figure 2 (default: 0)")
    parser.add_argument("--trace_n",    type=int, default=500,
                        help="Samples shown per chain in figure 3 (default: 500)")
    args = parser.parse_args()
    main(args.results, args.output_dir, args.mad_burnin, args.trace_n)
