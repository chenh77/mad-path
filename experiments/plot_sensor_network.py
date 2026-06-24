#!/usr/bin/env python3
"""
Publishable 3-panel figure for the sensor network localisation experiment.

Panels:
  (a) True network topology — anchor positions, sensor true positions, and
      observed distance links.
  (b) MAD-path posterior — per-sensor sample clouds, true positions, anchors.
  (c) APT-DEO posterior — same layout.

Usage:
    python experiments/plot_sensor_network.py \
        --result experiments/results/sensor_network.npz \
        --output experiments/plots/sensor_network.pdf
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
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

# Paul Tol "muted" qualitative palette — colorblind-safe, publication-standard
_SENSOR_COLORS = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
]
_TRUE_EDGE     = "black"
_LINK_COLOR    = "lightgray"   # light gray for observed distance links
_N_PLOT        = 2000        # max posterior samples to show per method
_MAD_COLOR     = "darkgreen"   # MAD-path
_APT1_COLOR    = "steelblue"   # chain 1 (init=zeros)
_APT2_COLOR    = "mediumpurple"  # chain 2 (init=ones)


def _load_results(result_path: str) -> dict:
    return dict(np.load(result_path, allow_pickle=True))


def _posterior_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    true_positions: np.ndarray,
    anchors: np.ndarray,
    title: str,
    *,
    rng: np.random.Generator,
    n_plot: int = _N_PLOT,
) -> None:
    """Draw per-sensor posterior clouds, true positions, and anchors."""
    valid = np.isfinite(samples).all(axis=1)
    samps = samples[valid]

    n = min(n_plot, len(samps))
    idx = rng.choice(len(samps), size=n, replace=False)
    pos_samps = samps[idx].reshape(-1, 8, 2) + 0.5   # (n, 8, 2) — uncentre to [0,1]^2

    for k in range(8):
        ax.scatter(
            pos_samps[:, k, 0], pos_samps[:, k, 1],
            s=3, alpha=0.25, color=_SENSOR_COLORS[k],
            rasterized=True, linewidths=0,
        )

    # True positions (hollow black circles)
    ax.scatter(
        true_positions[:, 0], true_positions[:, 1],
        s=20, facecolor="none", edgecolor=_TRUE_EDGE,
        linewidth=1.3, zorder=6,
    )

    # Known anchors (black stars)
    ax.scatter(
        anchors[:, 0], anchors[:, 1],
        marker="*", s=32, color="black",
        zorder=6,
    )

    _style_network_ax(ax, title)


def _style_network_ax(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$", labelpad=2)
    ax.set_ylabel("$z_2$", labelpad=2)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(length=3)
    ax.set_axisbelow(True)


def make_figure(data: dict, output: str) -> None:
    true_positions    = data["true_positions"]     # (8, 2)
    anchors           = data["anchors"]            # (3, 2)
    obs_pairs         = data["obs_pairs"]          # (M, 2)
    mad_samples       = data["mad_samples"]        # (N, 16)
    apt_deo_samples_1 = data["apt_deo_samples_1"] # (N, 16) full chain incl. burn-in
    apt_deo_samples_2 = data["apt_deo_samples_2"] # (N, 16) full chain incl. burn-in
    burnin_apt        = int(data["burnin_apt"])

    # Drop burn-in for posterior panels
    apt_post_1 = apt_deo_samples_1[burnin_apt:]
    apt_post_2 = apt_deo_samples_2[burnin_apt:]

    all_positions = np.concatenate([true_positions, anchors], axis=0)  # (11, 2)

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.0))
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.16, top=0.90, wspace=0.05)

    # ── Panel (a): true network topology ─────────────────────────────────────
    ax = axes[0]

    # Observed distance links
    for i, j in obs_pairs:
        ax.plot(
            [all_positions[i, 0], all_positions[j, 0]],
            [all_positions[i, 1], all_positions[j, 1]],
            color=_LINK_COLOR, lw=1.0, zorder=1,
        )

    # Unknown sensor true positions (hollow black circles with labels)
    for k in range(8):
        ax.scatter(
            true_positions[k, 0], true_positions[k, 1],
            s=20, facecolor="none", edgecolor="black",
            linewidth=1.3, zorder=5,
        )
        ax.text(
            true_positions[k, 0] - 0.02, true_positions[k, 1],
            f"$S${k+1}", fontsize=7, color="black", zorder=7,
            ha="right", va="bottom",
        )

    # Known anchors (black stars with labels)
    for m in range(len(anchors)):
        ax.scatter(
            anchors[m, 0], anchors[m, 1],
            marker="*", s=40, color="black",
            zorder=5,
        )

    _style_network_ax(ax, "Observed distances")

    # ── Panel (b): MAD-path posterior ─────────────────────────────────────────
    _posterior_panel(
        axes[1], mad_samples, true_positions, anchors,
        "MAD-Path",
        rng=rng,
    )

    # ── Panel (c): APT-DEO chain 1 (post-burnin, init=zeros) ─────────────────
    _posterior_panel(
        axes[2], apt_post_1, true_positions, anchors,
        r"APT-DEO (init = $\mathbf{0}_d$)",
        rng=rng,
    )

    # ── Panel (d): APT-DEO chain 2 (post-burnin, init=ones) ──────────────────
    _posterior_panel(
        axes[3], apt_post_2, true_positions, anchors,
        r"APT-DEO (init = $\mathbf{1}_d$)",
        rng=rng,
    )

    # Shared legend for panels (b)–(d): sensor colours + symbols
    sensor_patches = [
        mpatches.Patch(color=_SENSOR_COLORS[k], label=f"S{k+1}") for k in range(8)
    ]
    sensor_patches += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=_TRUE_EDGE, markeredgewidth=1.3,
               markersize=7, label="True pos."),
        Line2D([0], [0], marker="*", color="black", markersize=9, label="Anchor",
               linestyle="none"),
    ]
    fig.legend(
        handles=sensor_patches,
        loc="lower center",
        ncol=10,
        fontsize=9,
        framealpha=0,
        # edgecolor="gray",
        bbox_to_anchor=(0.5, -0.15),
        handlelength=1.2,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()
    plt.close(fig)


def make_figure_traces(data: dict, output: str) -> None:
    """Burn-in trace plots for MAD-path and both APT-DEO chains (4×4 grid).

    Layout — columns: x, y, x, y; rows: odd/even sensor pairs
      col 0–1: odd sensors  S1, S3, S5, S7  (x and y)
      col 2–3: even sensors S2, S4, S6, S8  (x and y)
    """
    mad_samples       = data["mad_samples"]
    apt_deo_samples_1 = data["apt_deo_samples_1"]
    apt_deo_samples_2 = data["apt_deo_samples_2"]
    burnin_apt        = int(data["burnin_apt"])

    # First burnin_apt rows approximate the burn-in phase
    mad_burn   = mad_samples[:burnin_apt]
    apt_burn_1 = apt_deo_samples_1[:burnin_apt]
    apt_burn_2 = apt_deo_samples_2[:burnin_apt]

    n_sensors = apt_burn_1.shape[1] // 2   # 8 sensors, 2 coords each
    n_rows = n_sensors // 2                # 4 row-pairs: (S1/S2), (S3/S4), (S5/S6), (S7/S8)
    n_cols = 4                             # x, y, x, y

    methods = [
        ("MAD-Path",         mad_burn,   _MAD_COLOR,  0.8),
        (r"APT-DEO (init = $\mathbf{0}_d$)", apt_burn_1, _APT1_COLOR, 0.7),
        (r"APT-DEO (init = $\mathbf{1}_d$)", apt_burn_2, _APT2_COLOR, 0.4),
    ]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 1.4 * n_rows), sharey=False)

    for r in range(n_rows):
        for c in range(n_cols):
            side       = c // 2              # 0 → left cols (odd sensors), 1 → right cols (even sensors)
            dim_idx    = c % 2               # 0 → x-coord, 1 → y-coord
            sensor_idx = 2 * r + side        # 0-indexed: 0=S1,1=S2,2=S3,...
            coord      = "z_1" if dim_idx == 0 else "z_2"
            d          = 2 * sensor_idx + dim_idx
            ax         = axes[r, c]
            ax.set_axisbelow(True)
            for name, chain, color, alpha in methods:
                ax.plot(chain[:, d] + 0.5, lw=0.6, alpha=alpha, color=color,
                        label=name if (r == 0 and c == 0) else None,
                        rasterized=True)
            ax.set_title(f"$S_{{{sensor_idx+1},{coord}}}$", fontsize=9, pad=2)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
                     framealpha=0, bbox_to_anchor=(0.5, -0.03))
    for lh in leg.get_lines():
        lh.set_linewidth(2.5)
    fig.supxlabel(f"iteration", fontsize=9, y=0.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _default_res = os.path.join(_script_dir, "results", "sensor_network.npz")
    _default_out = os.path.join(_script_dir, "plots")
    parser = argparse.ArgumentParser(description="Sensor network figures")
    parser.add_argument("--result", type=str, default=_default_res, help="Path to sensor_network.npz")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(_default_out, "sensor_network.pdf"),
    )
    parser.add_argument(
        "--output_traces", type=str,
        default=os.path.join(_default_out, "sensor_network_traces.pdf"),
    )
    args = parser.parse_args()

    data = _load_results(args.result)
    make_figure(data, args.output)
    make_figure_traces(data, args.output_traces)
