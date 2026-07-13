#!/usr/bin/env python3
"""
Publication-quality sampler comparison figure for the synthetic target.

3-row layout (one column per sampler):
  Row 0: x1–x2 scatter (dot plot), true mode centers marked with ×
  Row 1: x1 marginal histogram vs true distribution
  Row 2: x1–x2 trajectory trace, colored by iteration (light → dark)

Usage:
    # First generate results:
    python run_samplers_synthetic.py [--seed 0]
    # Then plot:
    python plot_sampler_comparison_synthetic.py [--results results/synthetic_comparison.npz]
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


# ── Load results ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NPZ = os.path.join(_SCRIPT_DIR, "results", "synthetic_comparison.npz")

parser = argparse.ArgumentParser()
parser.add_argument("--results", default=_DEFAULT_NPZ)
parser.add_argument("--output",  default=None,
                    help="Output path stem (default: plots/skew-particle-trace)")
args, _ = parser.parse_known_args()

data = np.load(args.results)

true_samples = data["true_samples"]    # (N_true, D)
target_mus   = data["target_mus"]      # (K, D) — true mode centers

SAMPLERS = [
    ("MAD-Path", data["mad_path_samples"],  None),
    ("APT",      data["apt_samples"],       None),
    ("APT-DEO",  data["apt_deo_samples"],   None),
    ("TT",       data["tt_samples"],        None),
    ("AIS",      data["ais_samples"],       data["ais_log_weights"]),
    ("SMC",      data["smc_samples"],       data["smc_log_weights"]),
    ("UD",       data["dds_samples"],       None),
]


# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "axes.grid":          True,
    "grid.linewidth":     0.35,
    "grid.color":         "#cccccc",
    "grid.alpha":         0.7,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

SCATTER_COLOR = "#2C4F90"    # muted blue for sampler dot plots
TRUE_COLOR    = "#C44E52"    # muted red for true reference / mode markers
HIST_SAMP     = "#FF7F0E"    # orange for sampler histogram bars
HIST_TRUE     = "#4878D0"    # steelblue for true histogram bars
MODE_COLOR    = TRUE_COLOR
STRIP_COLOR   = "#E8E8E8"
STRIP_EDGE    = "#aaaaaa"
STRIP_H       = 0.22         # strip height in axes-coord units


# ── Pre-process samples ───────────────────────────────────────────────────────
def _valid(samps):
    return np.asarray(samps)[np.isfinite(np.asarray(samps)).all(axis=1)]

def _imp_weights(log_w):
    lw = np.asarray(log_w).astype(float)
    w  = np.exp(lw - lw.max())
    return w / w.sum()

samples_list = []
weights_list = []
for _, samps, log_w in SAMPLERS:
    s = _valid(samps)
    samples_list.append(s)
    weights_list.append(_imp_weights(log_w) if log_w is not None else None)

# Per-column y-limits for IS sorted-weight plots (log scale)
_is_ylims = {}
for _col, _w in enumerate(weights_list):
    if _w is not None:
        _w_pos = _w[_w > 0]
        _is_ylims[_col] = (float(_w_pos.min()) * 0.5, float(_w.max()) * 2.0)


# ── Shared axis limits ────────────────────────────────────────────────────────
all_x1 = np.concatenate([s[:, 0] for s in samples_list] + [true_samples[:, 0]])
all_x2 = np.concatenate([s[:, 1] for s in samples_list] + [true_samples[:, 1]])

x1_lim = (float(np.percentile(all_x1, 0.1)), float(np.percentile(all_x1, 99.9)))
x2_lim = (float(np.percentile(all_x2, 0.1)), float(np.percentile(all_x2, 99.9)))

# histogram bin edges for x1, shared across all columns
N_BINS = 60
hist_edges = np.linspace(x1_lim[0], x1_lim[1], N_BINS + 1)

# true x1 histogram (reference)
true_hist, _ = np.histogram(true_samples[:, 0], bins=hist_edges, density=True)
hist_ymax = float(true_hist.max()) * 1.4


# ── Helper: ggplot2-style strip ───────────────────────────────────────────────
def _add_strip(ax, label: str) -> None:
    rect = Rectangle(
        (0, 1.0), 1.0, STRIP_H,
        transform=ax.transAxes, clip_on=False,
        facecolor=STRIP_COLOR, edgecolor=STRIP_EDGE, linewidth=0.5,
    )
    ax.add_patch(rect)
    ax.text(
        0.5, 1.0 + STRIP_H / 2, label,
        transform=ax.transAxes, clip_on=False,
        ha="center", va="center", fontsize=8,
    )


# ── Figure layout ─────────────────────────────────────────────────────────────
N_COL = len(SAMPLERS)
fig = plt.figure(figsize=(1.4 * N_COL + 0.86, 5.4))
gs  = gridspec.GridSpec(
    3, N_COL + 2,   # spacers at GridSpec cols 4 (MCMC/AIS) and 6 (AIS/SMC)
    figure=fig,
    top=0.86, bottom=0.09, left=0.07, right=0.995,
    hspace=0.55, wspace=0.15,
    width_ratios=[1, 1, 1, 1, 0.25, 1, 0.01, 1, 1],
)


# ── Draw each column ──────────────────────────────────────────────────────────
for col, (name, _, _) in enumerate(SAMPLERS):
    s     = samples_list[col]
    w     = weights_list[col]
    show_y = (col == 0)
    # spacers at GridSpec cols 4 (MCMC/AIS gap) and 6 (AIS/SMC gap)
    if col < 4:
        gs_col = col
    elif col == 4:
        gs_col = 5
    elif col == 5:
        gs_col = 7
    else:
        gs_col = 8

    # ── Row 0: x1–x2 scatter ─────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0, gs_col])

    # subsample for speed if very many particles
    n_show = min(len(s), 1000)
    idx = np.random.default_rng(0).choice(len(s), n_show, replace=False)
    ax_sc.scatter(
        s[idx, 0], s[idx, 1],
        s=1.5, alpha=0.35, color=SCATTER_COLOR, linewidths=0, rasterized=True,
    )
    ax_sc.scatter(
        target_mus[:, 0], target_mus[:, 1],
        s=14, marker="x", color=MODE_COLOR, linewidths=0.8, zorder=4,
    )

    ax_sc.set_xlim(*x1_lim)
    ax_sc.set_ylim(*x2_lim)
    ax_sc.set_xlabel(r"$x_1$", labelpad=2)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    if show_y:
        ax_sc.set_ylabel(r"$x_2$", labelpad=2)
    else:
        ax_sc.set_yticklabels([])
        ax_sc.tick_params(axis="y", left=False)
        ax_sc.spines["left"].set_visible(False)

    _add_strip(ax_sc, name)

    # ── Row 1: x1 histogram vs true ───────────────────────────────────────────
    ax_h = fig.add_subplot(gs[1, gs_col])

    # true x1 (steelblue, transparent)
    ax_h.bar(
        hist_edges[:-1], true_hist,
        width=np.diff(hist_edges), align="edge",
        color=HIST_TRUE, alpha=0.35, linewidth=0,
    )
    # sampler x1 (orange, semi-transparent)
    samp_hist, _ = np.histogram(s[:, 0], bins=hist_edges, density=True, weights=w)
    ax_h.bar(
        hist_edges[:-1], samp_hist,
        width=np.diff(hist_edges), align="edge",
        color=HIST_SAMP, alpha=0.65, linewidth=0,
    )

    ax_h.set_xlim(*x1_lim)
    ax_h.set_ylim(0, hist_ymax)
    ax_h.set_xlabel(r"$x_1$", labelpad=2)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    if show_y:
        ax_h.set_ylabel("density", labelpad=2, fontsize=9)
    else:
        ax_h.set_yticklabels([])
        ax_h.tick_params(axis="y", left=False)
        ax_h.spines["left"].set_visible(False)

    # ── Row 2: trace (MCMC) or sorted weight curve (IS) ─────────────────────
    ax_t = fig.add_subplot(gs[2, gs_col])

    if w is not None:
        # IS method — sorted weight curve on log scale
        w_sorted = np.sort(w)[::-1]
        ess      = 1.0 / float(np.sum(w ** 2))
        ax_t.semilogy(w_sorted, color=SCATTER_COLOR, lw=0.8)
        ax_t.axhline(1.0 / len(w), color=TRUE_COLOR, lw=0.8, ls="--", zorder=3)
        ax_t.set_ylim(*_is_ylims[col])
        ax_t.set_xlim(0, len(w) - 1)
        ax_t.set_xlabel("particle rank", labelpad=2)
        ax_t.text(0.95, 0.95, f"ESS={ess:.0f}", transform=ax_t.transAxes,
                  ha="right", va="top", fontsize=9)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        # AIS gets the y-label; SMC shows ticks but no label
        if col == 4:
            ax_t.set_ylabel("weight", labelpad=2, fontsize=9)
        elif col != 5:
            ax_t.set_yticklabels([])
            ax_t.tick_params(axis="y", left=False)
            ax_t.spines["left"].set_visible(False)
    elif name == "UD":
        ax_t.axis("off")
    else:
        # MCMC method — 2-D trajectory trace
        n = len(s)
        stride = max(1, n // min(n, 800))
        st = s[::stride, :2]
        ax_t.plot(st[:, 0], st[:, 1],
                  color=SCATTER_COLOR, lw=0.4, alpha=0.7, rasterized=True)
        ax_t.scatter(
            target_mus[:, 0], target_mus[:, 1],
            s=14, marker="x", color=MODE_COLOR, linewidths=0.8, zorder=4,
        )
        ax_t.set_xlim(*x1_lim)
        ax_t.set_ylim(*x2_lim)
        ax_t.set_xlabel(r"$x_1$", labelpad=2)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        if show_y:
            ax_t.set_ylabel(r"$x_2$", labelpad=2, fontsize=9)
        else:
            ax_t.set_yticklabels([])
            ax_t.tick_params(axis="y", left=False)
            ax_t.spines["left"].set_visible(False)


# ── Save ──────────────────────────────────────────────────────────────────────
if args.output is None:
    out_stem = os.path.join(_SCRIPT_DIR, "plots", "skew-particle-trace")
else:
    out_stem = args.output

os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", dpi=300)
print(f"Saved {out_stem}.pdf")
plt.show()
