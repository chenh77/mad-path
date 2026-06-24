#!/usr/bin/env python3
"""
Figures for the fetal-deaths T × N sensitivity sweep (paper Figures 7-8).

Two modes:
  --mode lines    Line plot of acceptance rate / inter-mode crossings vs. N,
                   one curve per T, median ± IQR across seeds.  Loads
                   seed_*.npz files produced by run_fetal_deaths.py --mode run.
  --mode traces   Runs four fresh MAD-path chains at representative (T, N)
                   settings and plots trace plots of the unconstrained
                   parameters.  Loads the shared state produced by
                   run_fetal_deaths.py --mode prepare.

Usage (from the repo root):
    python experiments/plot_fetal_deaths.py --mode lines \\
        --results_dir experiments/results/fetal_deaths/ \\
        --output      experiments/plots/fetal_deaths.pdf
    python experiments/plot_fetal_deaths.py --mode traces \\
        --shared_file experiments/results/fetal_deaths_shared.pkl \\
        --seed        0 \\
        --output      experiments/plots/fetal_deaths_traces.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from target.fetal_deaths import make_fetal_deaths_target
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.mcmc.mad_path import run_nonequil_mad_path

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
    "grid.linewidth":    0.4,
    "grid.color":        "#cccccc",
    "grid.alpha":        0.7,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# Must match run_fetal_deaths.py
T_VALS  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]   # sampling horizons (7.0 = T_TRAIN ★)
N_VALS  = [10, 20, 30, 40, 50, 60, 70, 100, 150, 300]
T_TRAIN = 7.0

METRICS = [
    ("accept_rate",      "Acceptance rate"),
    ("crossings_per_1k", "Inter-mode crossings per 1000 steps"),
]

# Blues palette from light to dark; T_TRAIN=7.0 gets darkest shade + dashes
T_COLORS = {
    1.0:  "#c6dbef",
    2.0:  "#9ecae1",
    3.0:  "#6baed6",
    4.0:  "#4292c6",
    5.0:  "#2171b5",
    6.0:  "#08519c",
    7.0:  "#08306b",   # training T — shown with dashes
}

# ── "traces" mode: must match run_fetal_deaths.py's prepare phase ─────────────
REF_STD     = 1.5
N_PARTICLES = 250
DDS_HIDDEN  = 64
SIGMA       = 1.0

NUM_STEPS = 10_000
BURNIN    = 5_000

# Representative (T_final, n_diffusion_steps) configurations to trace
CONFIGS = [
    (3.0,  30),
    (3.0, 150),
    (7.0,  30),
    (7.0, 150),
]

UNC_NAMES = [
    r"$\tilde{\gamma}$",
    r"$\tilde{\mu}$",
    r"$\tilde{\theta}$",
    r"$\tilde{\nu}$",
]

ROW_COLORS = ["#555555", "#555555", "#555555", "#555555"]


# ═══════════════════════════════════════════════════════════════════════════════
# "lines" mode — T × N sweep over seeds
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(results_dir: str) -> dict[str, np.ndarray]:
    """Return arrays[metric] of shape (n_seeds, n_T, n_N).

    Also verifies that T_VALS and N_VALS in files match the module constants.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"No seed_*.npz files found in {results_dir}")
    print(f"Loading {len(files)} seed files from {results_dir}")

    arrays: dict[str, list] = {m: [] for m, _ in METRICS}

    for path in files:
        d = np.load(path, allow_pickle=True)
        file_T = list(np.round(d["T_vals"].astype(float), 4))
        file_N = list(d["N_vals"].astype(int))

        for metric, _ in METRICS:
            if metric not in d:
                print(f"  Warning: '{metric}' missing in {os.path.basename(path)}")
                continue
            arr = np.asarray(d[metric])   # (n_T_file, n_N_file)

            # Align to canonical T_VALS × N_VALS (handles partial files)
            grid = np.full((len(T_VALS), len(N_VALS)), np.nan)
            for ti, T in enumerate(T_VALS):
                if round(T, 4) in file_T:
                    fi = file_T.index(round(T, 4))
                    for ni, N in enumerate(N_VALS):
                        if N in file_N:
                            grid[ti, ni] = arr[fi, file_N.index(N)]
            arrays[metric].append(grid)

    return {m: np.stack(arrays[m]) for m in arrays if arrays[m]}
    # shape: (n_seeds, n_T, n_N)


def plot_lines(arrays: dict, output: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    x = np.array(N_VALS)

    for ax, (metric, ylabel) in zip(axes, METRICS):
        for ti, T in enumerate(T_VALS):
            arr  = arrays[metric][:, ti, :]   # (n_seeds, n_N)
            med  = np.nanmedian(arr, axis=0)
            p25  = np.nanpercentile(arr, 25, axis=0)
            p75  = np.nanpercentile(arr, 75, axis=0)
            c    = T_COLORS[T]
            ls   = "--" if T == T_TRAIN else "-"
            lbl  = f"$T=${T}" + (" ★" if T == T_TRAIN else "")

            ax.plot(x, med, color=c, linewidth=1.8,
                    linestyle=ls, marker="o", markersize=4,
                    label=lbl)
            ax.fill_between(x, p25, p75, color=c, alpha=0.18)

        ax.set_xscale("log")
        ax.set_xticks(N_VALS)
        ax.set_xticklabels([str(n) for n in N_VALS])
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=7.5, framealpha=0.85, loc="best", ncols=2)
    fig.tight_layout(pad=2.0, w_pad=4.0)
    _save(fig, output)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved → {path}")
    plt.show()


def print_summary(arrays: dict) -> None:
    print(f"\n{'T':>6}  {'N':>5}  {'metric':<32}  "
          f"{'median':>8}  {'mean':>8}  {'std':>8}  n")
    print("-" * 72)
    for ti, T in enumerate(T_VALS):
        for ni, N in enumerate(N_VALS):
            for metric, ylabel in METRICS:
                vals = arrays[metric][:, ti, ni]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                print(f"{T:6.1f}  {N:5d}  {ylabel:<32}  "
                      f"{np.median(vals):8.4f}  {np.mean(vals):8.4f}  "
                      f"{np.std(vals):8.4f}  {len(vals)}")
        print()


def main_lines(results_dir: str, output: str) -> None:
    arrays = load_results(results_dir)
    print_summary(arrays)
    plot_lines(arrays, output)


# ═══════════════════════════════════════════════════════════════════════════════
# "traces" mode — fresh chains at representative (T, N) settings
# ═══════════════════════════════════════════════════════════════════════════════

def run_chain(
    T_final: float,
    N: int,
    seed: int,
    log_prob_scaled,
    dds,
    params,
    coord_scale,
    shift,
    sigma_vec,
    x0_s,
) -> np.ndarray:
    """Return post-burnin chain in original unconstrained space, shape (n_steps, 4)."""
    key = jr.PRNGKey(seed * 1000 + int(T_final * 10) + N)
    chain_s, accept = run_nonequil_mad_path(
        log_target=log_prob_scaled,
        dds_sampler=dds,
        params=params,
        sigma=sigma_vec,
        T_final=T_final,
        initial_state=x0_s,
        num_steps=NUM_STEPS,
        n_diffusion_steps=N,
        rng_key=key,
        with_correction=True,
        burnin=BURNIN,
    )
    chain_unc = np.asarray(chain_s) * np.asarray(coord_scale)[None, :] + np.asarray(shift)[None, :]
    print(f"  T={T_final}  N={N:4d}  accept={accept:.3f}  chain={chain_unc.shape}")
    return chain_unc


def main_traces(shared_file: str, seed: int, output: str) -> None:
    if not os.path.exists(shared_file):
        raise FileNotFoundError(
            f"Shared state not found: {shared_file}\n"
            "Run `python experiments/run_fetal_deaths.py --mode prepare` first.")

    with open(shared_file, "rb") as f:
        saved = pickle.load(f)

    shift       = jnp.asarray(saved["shift"],       dtype=jnp.float32)
    coord_scale = jnp.asarray(saved["coord_scale"], dtype=jnp.float32)
    log_jac     = float(saved["log_jac"])
    means_s     = jnp.asarray(saved["means_s"],     dtype=jnp.float32)
    vars_s      = jnp.asarray(saved["vars_s"],      dtype=jnp.float32)
    weights_s   = jnp.asarray(saved["weights_np"],  dtype=jnp.float32)
    params      = saved["params"]
    x0_s        = jnp.asarray(saved["x0_s"],        dtype=jnp.float32)
    T_train     = float(saved["T_train"])
    N_train     = int(saved["N_train"])

    target = make_fetal_deaths_target(ref_std=REF_STD)
    D      = target.dim

    def log_prob_scaled(phi):
        return target.log_prob(phi * coord_scale + shift) + log_jac

    sigma_vec = jnp.full(D, SIGMA, dtype=jnp.float32)

    config = NonequilibriumDDSConfig(
        T_FINAL=T_train, D=D,
        n_particles=N_PARTICLES,
        hidden=DDS_HIDDEN,
        score_parametrization="base",
        detach_target_score=True,
        sigma=sigma_vec,
    )
    dds = make_nonequilibrium_dds_sampler(
        log_prob_scaled, means_s, vars_s, weights_s, config)

    print(f"Running chains  seed={seed}  D={D}")
    chains = []
    for T_final, N in CONFIGS:
        chains.append(run_chain(
            T_final, N, seed,
            log_prob_scaled, dds, params,
            coord_scale, shift, sigma_vec, x0_s,
        ))

    # ── Plot ──────────────────────────────────────────────────────────────────
    # rows = parameters, cols = (T_final, N) configurations
    n_rows, n_cols = D, len(CONFIGS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 1.6 * n_rows),
        squeeze=False,
    )

    steps = np.arange(chains[0].shape[0])

    # compute per-row y limits (same parameter, across all configs)
    row_ylims = []
    for row in range(n_rows):
        all_vals = np.concatenate([c[:, row] for c in chains])
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        row_ylims.append((lo - pad, hi + pad))

    for col, ((T_final, N), chain, color) in enumerate(zip(CONFIGS, chains, ROW_COLORS)):
        for row, name in enumerate(UNC_NAMES):
            ax = axes[row][col]
            ax.plot(steps, chain[:, row], lw=0.25, alpha=0.75, color=color)
            ax.set_xlim(0, len(steps))
            ax.set_ylim(*row_ylims[row])

            if row == 0:
                ax.set_title(f"$T={int(T_final)},\\ N={N}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(name, fontsize=9)

    fig.supxlabel("Iteration", fontsize=9)
    fig.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _RESULTS = os.path.join(_SCRIPT_DIR, "results")
    _PLOTS   = os.path.join(_SCRIPT_DIR, "plots")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lines", "traces"], default="lines")
    parser.add_argument("--results_dir", default=os.path.join(_RESULTS, "fetal_deaths"),
                        help="[lines] directory of seed_*.npz files")
    parser.add_argument("--shared_file", default=os.path.join(_RESULTS, "fetal_deaths_shared.pkl"),
                        help="[traces] shared state from run_fetal_deaths.py --mode prepare")
    parser.add_argument("--seed", type=int, default=0,
                        help="[traces] seed for the fresh MAD-path chains")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "lines":
        output = args.output or os.path.join(_PLOTS, "fetal_deaths.pdf")
        main_lines(args.results_dir, output)
    else:
        output = args.output or os.path.join(_PLOTS, "fetal_deaths_traces.pdf")
        main_traces(args.shared_file, args.seed, output)
