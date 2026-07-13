#!/usr/bin/env python3
"""
Publication-quality sampler comparison figure.

Row 1: horizontal KDE of v1^T x — density on x-axis (bottom), projection on y-axis
Row 2: trace plot of v1^T x
Both rows share the same y-range. Column titles in ggplot2-style gray strips.

Usage:
    # First generate results:
    python run_samplers.py [--seed 0]
    # Then plot:
    python plot_sampler_comparison.py [--results results/gmm_d20_comparison.npz]
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

# ── Target definition ─────────────────────────────────────────────────────────
D = 20
_TGT_MEANS = np.array([np.ones(D), -np.ones(D)])
_TGT_STD   = np.array([np.sqrt(0.5), np.sqrt(0.2)])
_TGT_W     = np.array([0.5, 0.5])
v1 = np.ones(D) / np.sqrt(D)

def _true_proj_pdf(x: np.ndarray) -> np.ndarray:
    """Analytic density of v1^T X (1-D GMM)."""
    pdf = np.zeros_like(x, dtype=float)
    for k in range(2):
        mu = float(_TGT_MEANS[k] @ v1)
        s  = _TGT_STD[k]
        pdf += _TGT_W[k] / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return pdf


# ── Load results ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NPZ = os.path.join(_SCRIPT_DIR, "results", "gmm_d20_comparison.npz")
_DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "plots", "mog-particle-trace")

parser = argparse.ArgumentParser()
parser.add_argument("--results", default=_DEFAULT_NPZ)
parser.add_argument("--output",  default=_DEFAULT_OUTPUT,
                    help="Output path stem (default: plots/mog-particle-trace)")
args, _ = parser.parse_known_args()

data = np.load(args.results)

SAMPLERS = [
    ("Mad-Path", data["mad_path_samples"],  None),
    ("APT",      data["apt_samples"],       None),
    ("APT-DEO",  data["apt_deo_samples"],   None),
    ("TT",       data["tt_samples"],        None),
    ("AIS",      data["ais_samples"],       data["ais_log_weights"]),
    ("SMC",      data["smc_samples"],       data["smc_log_weights"]),
    ("DS",      data["dds_samples"],       None),
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

KDE_COLOR   = "#333333"     # near-black curve, matching reference
FILL_COLOR  = "#333333"
TRUE_COLOR  = "#C44E52"     # muted red for true marginal
TRACE_COLOR = "#333333"
FILL_ALPHA  = 0.12
STRIP_COLOR = "#E8E8E8"     # gray strip background
STRIP_EDGE  = "#aaaaaa"
STRIP_H     = 0.22          # strip height in axes-coord units


# ── Projections ───────────────────────────────────────────────────────────────
projs, imp_weights = [], []
for _, samps, log_w in SAMPLERS:
    valid = np.isfinite(np.asarray(samps)).all(axis=1)
    proj  = np.asarray(samps)[valid] @ v1
    projs.append(proj)
    if log_w is not None:
        lw = np.asarray(log_w)[valid].astype(float)
        w  = np.exp(lw - lw.max())
        imp_weights.append(w / w.sum())
    else:
        imp_weights.append(None)


# ── Shared axis limits ────────────────────────────────────────────────────────
y_min = min(p.min() for p in projs)
y_max = max(p.max() for p in projs)
y_pad = (y_max - y_min) * 0.06
y_lim = (y_min - y_pad, y_max + y_pad)

yy       = np.linspace(y_lim[0], y_lim[1], 500)
true_pdf = _true_proj_pdf(yy)

# KDE for each column.
# Scott's rule uses n^{-1/5} * std(data), but std(data) is inflated (~4.5) by
# the inter-mode separation, giving bandwidth ~0.82 — much wider than the true
# mode stds (0.45, 0.71).  Fix: use a fixed absolute bandwidth matched to the
# narrower mode std instead.
# _KDE_BW = float(np.sqrt(_TGT_STD[0] ** 2 + _TGT_STD[1] ** 2) / 2)  # ~0.58
_KDE_BW = 0.2
kde_curves = []
for proj, w in zip(projs, imp_weights):
    bw_factor = _KDE_BW / float(proj.std())   # absolute bandwidth / data std
    kde = gaussian_kde(proj, bw_method=bw_factor, weights=w)
    kde_curves.append(kde(yy))

x_max = max(float(d.max()) for d in kde_curves)
x_max = max(x_max, float(true_pdf.max())) * 1.15

# Per-column y-limits for IS sorted-weight plots (log scale)
_is_ylims = {}
for _col, _w in enumerate(imp_weights):
    if _w is not None:
        _w_pos = _w[_w > 0]
        _is_ylims[_col] = (float(_w_pos.min()) * 0.5, float(_w.max()) * 2.0)


# ── Figure layout ─────────────────────────────────────────────────────────────
N_COL  = len(SAMPLERS)
fig    = plt.figure(figsize=(1.5 * N_COL + 1.11, 3.5))
gs     = gridspec.GridSpec(
    2, N_COL + 2,   # spacers at GridSpec cols 4 (MCMC/IS) and 7 (AIS/SMC)
    figure=fig,
    top=0.84, bottom=0.11, left=0.08, right=0.995,
    hspace=0.38, wspace=0.15,
    width_ratios=[1, 1, 1, 1, 0.25, 1, 0.01, 1, 1],
)

# Separate y-axis label per row (y positions computed from GridSpec geometry)
_label_kw = dict(va="center", rotation=90, fontsize=9, color="#333333")
fig.text(0.04, 0.687, r"$v_1^\top x$", **_label_kw)
fig.text(0.04, 0.263, r"$v_1^\top x$", **_label_kw)


def _add_strip(ax, label: str) -> None:
    """Draw a full-width gray strip above the axes, mimicking ggplot2 facet labels."""
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


# ── Draw each column ──────────────────────────────────────────────────────────
for col, (name, _, _) in enumerate(SAMPLERS):
    kde_d  = kde_curves[col]
    proj   = projs[col]
    show_y = (col == 0)
    # spacers at GridSpec cols 4 (MCMC/IS) and 7 (AIS/SMC)
    if col < 4:
        gs_col = col
    elif col == 4:
        gs_col = 5
    elif col == 5:
        gs_col = 7
    else:
        gs_col = 8

    # ── Row 0: horizontal KDE ─────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, gs_col])

    # Filled area under KDE
    ax_h.fill_betweenx(yy, 0, kde_d,
                       color=FILL_COLOR, alpha=FILL_ALPHA, linewidth=0)
    # KDE outline
    ax_h.plot(kde_d, yy, color=KDE_COLOR, lw=0.9)
    # True marginal (dashed red)
    ax_h.plot(true_pdf, yy, color=TRUE_COLOR, lw=1.0, ls="--", zorder=3)

    ax_h.set_ylim(*y_lim)
    ax_h.set_xlim(0, x_max)
    ax_h.set_xlabel("density", labelpad=2)

    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)
    if not show_y:
        ax_h.set_yticklabels([])
        ax_h.tick_params(axis="y", left=False)
        ax_h.spines["left"].set_visible(False)

    _add_strip(ax_h, name)

    # ── Row 1: trace plot (MCMC) or sorted weight curve (IS) ─────────────────
    ax_t = fig.add_subplot(gs[1, gs_col])

    if imp_weights[col] is not None:
        # IS method — sorted weight curve on log scale
        w        = imp_weights[col]
        w_sorted = np.sort(w)[::-1]
        ess      = 1.0 / float(np.sum(w ** 2))
        ax_t.semilogy(w_sorted, color=TRACE_COLOR, lw=0.8)
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
    elif name == "DS":
        # leave row 1 blank
        ax_t.axis("off")
    else:
        # MCMC method — trace plot
        ax_t.plot(proj, color=TRACE_COLOR, lw=0.35, alpha=0.85, rasterized=True)
        ax_t.set_ylim(*y_lim)
        ax_t.set_xlim(0, len(proj) - 1)
        n = len(proj)
        ax_t.set_xticks([0, n // 2, n - 1])
        ax_t.set_xticklabels(["0", str(n // 2), str(n)])
        ax_t.set_xlabel("iteration", labelpad=2)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        if not show_y:
            ax_t.set_yticklabels([])
            ax_t.tick_params(axis="y", left=False)
            ax_t.spines["left"].set_visible(False)


# ── Save ──────────────────────────────────────────────────────────────────────
out_stem = args.output
os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", dpi=300)
print(f"Saved {out_stem}.pdf")
plt.show()
