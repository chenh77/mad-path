#!/usr/bin/env python3
"""
Run MAD-Path, APT, and APT+DEO on the Bayesian GMM target (Old Faithful data).

Two-phase workflow (matches sweep_bayesian_gmm.sh):
  --mode prepare  Run AL, fit GMM, train DDS. Save shared state to --shared_file.
  --mode run      Load shared state, run MCMC chains for --seed, save seed_NNNN.npz.

Seed controls: APT_INIT_KEY = APT_RUN_KEY = MAD_KEY = jr.PRNGKey(seed).
AL and DDS training are deterministic (no seed) and shared across all runs.

Output per seed:
  seed_NNNN.npz with {apt,apt_deo,mad}_{mode1_weight,crossings_per_1k}
  compatible with plot_bayesian_gmm.py.

Usage (from the repo root):
    python experiments/run_bayesian_gmm.py --mode prepare --shared_file experiments/results/bgmm_shared.pkl
    python experiments/run_bayesian_gmm.py --mode run --seed 0 \\
        --shared_file experiments/results/bgmm_shared.pkl --output_dir experiments/results/bayesian_gmm/
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import numpy as np
from sklearn.cluster import KMeans

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from target.datasets import load_old_faithful
from target.bayesian_gmm import BayesianGMMTarget
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path
from sampler.mcmc.adaptive_parallel_tempering import adaptive_parallel_tempering

# ── APT / APT+DEO ─────────────────────────────────────────────────────────────
N_TEMPS_APT    = 50
N_ITER_APT     = 12000
BURNIN_APT     = 2000
APT_STEP       = 0.05
APT_REF_STD    = 1.0
APT_MALA_STEPS = 10
APT_GAMMA_BASE     = 9.2
APT_DEO_GAMMA_BASE = 0.25

# ── Annealed Langevin ─────────────────────────────────────────────────────────
N_AL       = 1000
N_TEMPS_AL = 5       # temperature levels
N_STEPS_AL = 1000    # ULA steps per level
AL_REF_STD = 0.0     # x0 = zeros (notebook: REF_STD = 0.)

# ── Mode finding ──────────────────────────────────────────────────────────────
MODE_GD_STEPS     = 20
MODE_GD_LR        = 1e-3
MODE_GRAD_CLIP    = 10.0
MODE_MERGE_RADIUS = 1.0
MIN_MODE_COUNT    = 10

# ── Nonequilibrium DDS ────────────────────────────────────────────────────────
T_FINAL            = 7.0
N_PARTICLES_TRAIN  = 500
N_ITER_TRAIN       = 200
N_DIFFUSION_TRAIN  = 200   # n_steps for training
N_DIFFUSION_SAMPLE = 100   # n_steps for generate_samples (notebook: n_steps=100)
DDS_LR             = 1e-2
DDS_SIGMA          = 1.0
DDS_HIDDEN         = 32

# ── MAD-Path ──────────────────────────────────────────────────────────────────
NUM_STEPS_MAD   = 10000
N_DIFFUSION_MAD = 200
N_SAMP_DDS      = 2000   # DDS particles used to pick initial state
MAD_INIT_IDX    = 2      # notebook: nonequil_samples_scaled[2]

# ── Fixed PRNG keys for shared (deterministic) phases ─────────────────────────
AL_KEY  = jr.PRNGKey(0)   # run_annealed_langevin(jr.PRNGKey(0))
DDS_KEY = jr.PRNGKey(1)   # ik, tk, sk = jr.split(jr.PRNGKey(1), 3)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_target():
    data_raw  = load_old_faithful()
    data_norm = (data_raw - data_raw.mean(0)) / data_raw.std(0)
    target    = BayesianGMMTarget(data_norm, K=2, ref_std=3.0)
    return target


def _compute_metrics(samples: np.ndarray, mode_centers: np.ndarray) -> dict:
    """Compute scalar metrics for one method's sample array."""
    d1    = np.sum((samples - mode_centers[0]) ** 2, axis=-1)
    d2    = np.sum((samples - mode_centers[1]) ** 2, axis=-1)
    mode1 = (d1 <= d2).astype(float)

    n                = len(mode1)
    mode1_weight     = float(np.mean(mode1))
    crossings        = float(np.sum(np.abs(np.diff(mode1))))
    crossings_per_1k = crossings / (n / 1000.0)

    return dict(
        mode1_weight=mode1_weight,
        crossings_per_1k=crossings_per_1k,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — prepare (run once)
# ═══════════════════════════════════════════════════════════════════════════════

def main_prepare(shared_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(shared_file)), exist_ok=True)

    target = _build_target()
    D      = target.dim
    print(f"BayesianGMM target: D={D}")

    # ── Annealed Langevin ─────────────────────────────────────────────────────
    print(f"\nAnnealed Langevin ({N_AL} particles, {N_TEMPS_AL} temps × {N_STEPS_AL} steps)...")
    t0 = time.time()

    betas = jnp.power(0.5, 1 - jnp.linspace(0, 1, N_TEMPS_AL))

    @jit
    def run_annealed_langevin(key):
        key, init_key = jr.split(key)
        x0            = AL_REF_STD * jr.normal(init_key, (N_AL, D))
        beta_schedule = jnp.repeat(betas, N_STEPS_AL)
        keys          = jr.split(key, len(beta_schedule))

        def step(x, inp):
            beta, k = inp
            score   = beta * vmap(jax.grad(target.log_prob))(x)
            noise   = jr.normal(k, x.shape)
            h       = 1e-4 / (beta + 1e-4)
            return x + h * score + jnp.sqrt(2 * h) * noise, None

        x_final, _ = jax.lax.scan(step, x0, (beta_schedule, keys))
        return x_final

    al_samples = np.asarray(run_annealed_langevin(AL_KEY))
    print(f"  Done in {time.time()-t0:.1f}s  shape={al_samples.shape}")

    # ── Mode finding ──────────────────────────────────────────────────────────
    print("\nMode finding (gradient ascent + clustering)...")
    t0 = time.time()

    al_jnp = jnp.array(al_samples, dtype=jnp.float32)

    @jit
    def ascend_batch(x0):
        def step(x, _):
            score = jnp.clip(vmap(jax.grad(target.log_prob))(x),
                             -MODE_GRAD_CLIP, MODE_GRAD_CLIP)
            return x + MODE_GD_LR * score, None
        x_final, _ = jax.lax.scan(step, x0, None, length=MODE_GD_STEPS)
        return x_final

    mode_endpoints = np.asarray(ascend_batch(al_jnp))
    mode_logps     = np.asarray(vmap(target.log_prob)(jnp.asarray(mode_endpoints)))
    finite         = np.isfinite(mode_logps) & np.isfinite(mode_endpoints).all(axis=1)
    if not np.all(finite):
        print(f"  Dropping {(~finite).sum()} non-finite endpoints")
    mode_endpoints = mode_endpoints[finite]
    mode_logps     = mode_logps[finite]

    order = np.argsort(-mode_logps)
    mode_centers, mode_counts = [], []
    for idx in order:
        x = mode_endpoints[idx]
        if not mode_centers:
            mode_centers.append(x); mode_counts.append(1); continue
        nearest = int(np.argmin(np.linalg.norm(np.stack(mode_centers) - x[None], axis=1)))
        if np.linalg.norm(mode_centers[nearest] - x) <= MODE_MERGE_RADIUS:
            mode_counts[nearest] += 1
        else:
            mode_centers.append(x); mode_counts.append(1)

    mode_centers = [c for c, n in zip(mode_centers, mode_counts) if n >= MIN_MODE_COUNT]
    mode_counts  = [n for n in mode_counts if n >= MIN_MODE_COUNT]
    K_FIT        = len(mode_centers)
    print(f"  Found {K_FIT} modes: counts={mode_counts}  ({time.time()-t0:.1f}s)")

    # ── K-means GMM fit + coordinate rescaling ────────────────────────────────
    print("\nK-means GMM fit + coordinate rescaling...")
    km = KMeans(n_clusters=K_FIT, init=np.stack(mode_centers), n_init=1)
    km.fit(al_samples)

    gmm_means, gmm_vars_diag, weights_list = [], [], []
    for k in range(K_FIT):
        members = al_samples[km.labels_ == k]
        gmm_means.append(km.cluster_centers_[k])
        gmm_vars_diag.append(np.diag(np.cov(members, rowvar=False)))
        weights_list.append(members.shape[0] / len(al_samples))
    gmm_means     = np.stack(gmm_means)
    gmm_vars_diag = np.stack(gmm_vars_diag)
    weights_np    = np.array(weights_list)

    shift = np.asarray(jnp.mean(al_jnp, axis=0))
    scale = np.asarray(jnp.min(jnp.sqrt(jnp.array(gmm_vars_diag)), axis=0))

    means_s = (gmm_means - shift) / scale
    vars_s  = gmm_vars_diag / scale ** 2

    print(f"  scaled abs-max per coord: {np.round(np.max(np.abs(means_s), axis=0), 2)}")

    # ── Nonequil DDS training ─────────────────────────────────────────────────
    print(f"\nNonequil DDS training "
          f"({N_PARTICLES_TRAIN} particles × {N_ITER_TRAIN} iters, T={T_FINAL})...")
    t_train = time.time()

    log_jac = float(np.sum(np.log(scale)))

    def scaled_logp(x):
        return target.log_prob(x * jnp.asarray(scale) + jnp.asarray(shift)) + log_jac

    _norm_weights = jnp.exp(
        jnp.log(jnp.array(weights_np))
        - jax.scipy.special.logsumexp(jnp.log(jnp.array(weights_np)))
    )

    nonequil_config = NonequilibriumDDSConfig(
        T_FINAL=T_FINAL, D=D, n_particles=N_PARTICLES_TRAIN,
        hidden=DDS_HIDDEN, score_parametrization="base", sigma=DDS_SIGMA,
    )
    nonequil_dds = make_nonequilibrium_dds_sampler(
        scaled_logp, means_s, vars_s, _norm_weights, nonequil_config,
    )

    ik, tk, _ = jr.split(DDS_KEY, 3)
    params, metrics = train(
        nonequil_dds.init_params(ik),
        nonequil_dds.make_pathwise_fns(n_steps=N_DIFFUSION_TRAIN),
        n_iter=N_ITER_TRAIN, lr=DDS_LR, grad_clip=10.0, log_every=20, key=tk,
    )
    train_time_s = time.time() - t_train
    print(f"  Final loss: {metrics['losses'][-1]:.4f}  ({train_time_s:.1f}s)")

    # ── Save shared state ─────────────────────────────────────────────────────
    shared = dict(
        shift=shift,
        scale=scale,
        means_s=means_s,
        vars_s=vars_s,
        weights_np=weights_np,
        mode_centers=np.stack(mode_centers),
        params=params,
        dds_losses=np.array(metrics["losses"]),
        train_time_s=train_time_s,
    )
    with open(shared_file, "wb") as f:
        pickle.dump(shared, f)
    print(f"\nShared state saved → {shared_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — run (one per seed)
# ═══════════════════════════════════════════════════════════════════════════════

def main_run(seed: int, shared_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seed_{seed:04d}.npz")

    # ── Load shared state ─────────────────────────────────────────────────────
    with open(shared_file, "rb") as f:
        shared = pickle.load(f)

    shift        = jnp.asarray(shared["shift"])
    scale        = jnp.asarray(shared["scale"])
    means_s      = shared["means_s"]
    vars_s       = shared["vars_s"]
    weights_np   = shared["weights_np"]
    mode_centers = shared["mode_centers"]
    params       = shared["params"]
    train_time_s = float(shared["train_time_s"])

    target = _build_target()
    D      = target.dim

    def scaled_logp(x):
        return target.log_prob(x * scale + shift) + jnp.sum(jnp.log(scale))

    _norm_weights = jnp.exp(
        jnp.log(jnp.array(weights_np))
        - jax.scipy.special.logsumexp(jnp.log(jnp.array(weights_np)))
    )
    nonequil_config = NonequilibriumDDSConfig(
        T_FINAL=T_FINAL, D=D, n_particles=N_PARTICLES_TRAIN,
        hidden=DDS_HIDDEN, score_parametrization="base", sigma=DDS_SIGMA,
    )
    nonequil_dds = make_nonequilibrium_dds_sampler(
        scaled_logp, means_s, vars_s, _norm_weights, nonequil_config,
    )

    # ── Seed-dependent PRNG keys ──────────────────────────────────────────────
    # APT_INIT_KEY = APT_RUN_KEY = MAD_KEY = PRNGKey(seed)
    chain_key    = jr.PRNGKey(seed)
    apt_init_key = chain_key
    apt_run_key  = chain_key
    mad_key      = chain_key   # same as apt_init_key

    apt_schedule    = jnp.power(0.05, jnp.linspace(0, 1, N_TEMPS_APT))
    apt_init_states = APT_REF_STD * jr.normal(apt_init_key, (N_TEMPS_APT, D))

    # ── APT (no DOE) ──────────────────────────────────────────────────────────
    print(f"[seed={seed}] APT...")
    t0 = time.time()
    apt_samples, apt_diag = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=apt_init_states,
        initial_inverse_temperature_schedule=apt_schedule,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        initial_step_size=APT_STEP,
        num_mala_steps=APT_MALA_STEPS,
        rng_key=apt_run_key,
        use_doe=False,
        gamma_temperature_base=APT_GAMMA_BASE,
        gamma_mala_base=APT_GAMMA_BASE,
    )
    apt_samples = np.asarray(apt_samples)
    apt_time_s  = time.time() - t0
    print(f"  swap_accept={float(apt_diag['swap_acceptance_rate']):.3f}  "
          f"n={apt_samples.shape[0]}  ({apt_time_s:.1f}s)")

    # ── APT+DEO ───────────────────────────────────────────────────────────────
    print(f"[seed={seed}] APT+DEO...")
    t0 = time.time()
    apt_deo_samples, apt_deo_diag = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=apt_init_states,
        initial_inverse_temperature_schedule=apt_schedule,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        initial_step_size=APT_STEP,
        num_mala_steps=APT_MALA_STEPS,
        rng_key=apt_run_key,
        use_doe=True,
        gamma_temperature_base=APT_DEO_GAMMA_BASE,
        gamma_mala_base=APT_DEO_GAMMA_BASE,
    )
    apt_deo_samples = np.asarray(apt_deo_samples)
    apt_deo_time_s  = time.time() - t0
    print(f"  swap_accept={float(apt_deo_diag['swap_acceptance_rate']):.3f}  "
          f"n={apt_deo_samples.shape[0]}  ({apt_deo_time_s:.1f}s)")

    # ── MAD-Path ───────────────────────────────────────────────────────────────
    print(f"[seed={seed}] MAD-Path...")
    t0 = time.time()
    _, _, sk = jr.split(DDS_KEY, 3)
    dds_samples_s, _ = nonequil_dds.generate_samples(
        params, sk, n_particles=N_SAMP_DDS, n_steps=N_DIFFUSION_SAMPLE,
    )
    init_mad_s = jnp.array(np.asarray(dds_samples_s)[MAD_INIT_IDX])
    chain_mad_s, acc_mad = run_nonequil_mad_path(
        log_target=scaled_logp,
        dds_sampler=nonequil_dds,
        params=params,
        sigma=DDS_SIGMA,
        T_final=T_FINAL,
        initial_state=init_mad_s,
        num_steps=NUM_STEPS_MAD,
        n_diffusion_steps=N_DIFFUSION_MAD,
        n_mala_steps=0,
        mala_step_size=1e-1,
        rng_key=mad_key,
        with_correction=True,
    )
    mad_samples = (np.asarray(chain_mad_s) * np.asarray(scale)[None]
                   + np.asarray(shift)[None])
    mad_time_s  = time.time() - t0 + train_time_s
    print(f"  accept={acc_mad:.3f}  n={mad_samples.shape[0]}  ({mad_time_s:.1f}s)")

    # ── Compute metrics ───────────────────────────────────────────────────────
    apt_m     = _compute_metrics(apt_samples,     mode_centers)
    apt_deo_m = _compute_metrics(apt_deo_samples, mode_centers)
    mad_m     = _compute_metrics(mad_samples,     mode_centers)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        seed=seed,
        mode_centers=mode_centers,
        # raw samples
        apt_samples=apt_samples,
        apt_deo_samples=apt_deo_samples,
        mad_samples=mad_samples,
        # metrics expected by plot_bayesian_gmm.py
        apt_mode1_weight=apt_m["mode1_weight"],
        apt_crossings_per_1k=apt_m["crossings_per_1k"],
        apt_deo_mode1_weight=apt_deo_m["mode1_weight"],
        apt_deo_crossings_per_1k=apt_deo_m["crossings_per_1k"],
        mad_mode1_weight=mad_m["mode1_weight"],
        mad_crossings_per_1k=mad_m["crossings_per_1k"],
        # diagnostics
        acc_mad=acc_mad,
        apt_swap_rate=float(apt_diag["swap_acceptance_rate"]),
        apt_deo_swap_rate=float(apt_deo_diag["swap_acceptance_rate"]),
        # wall-clock time per method (seconds)
        apt_time_s=apt_time_s,
        apt_deo_time_s=apt_deo_time_s,
        mad_time_s=mad_time_s,
    )
    print(f"  Saved → {out_path}")
    print(f"  APT:     mode1={apt_m['mode1_weight']:.3f}  "
          f"cross/1k={apt_m['crossings_per_1k']:.2f}")
    print(f"  APT+DEO: mode1={apt_deo_m['mode1_weight']:.3f}  "
          f"cross/1k={apt_deo_m['crossings_per_1k']:.2f}")
    print(f"  MAD:     mode1={mad_m['mode1_weight']:.3f}  "
          f"cross/1k={mad_m['crossings_per_1k']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _RESULTS = os.path.join(_SCRIPT_DIR, "results")
    parser   = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["prepare", "run"], required=True)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--shared_file", type=str,
                        default=os.path.join(_RESULTS, "bayesian_gmm_shared.pkl"))
    parser.add_argument("--output_dir",  type=str,
                        default=os.path.join(_RESULTS, "bayesian_gmm"))
    args = parser.parse_args()

    if args.mode == "prepare":
        main_prepare(args.shared_file)
    else:
        main_run(args.seed, args.shared_file, args.output_dir)
