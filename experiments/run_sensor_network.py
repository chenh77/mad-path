#!/usr/bin/env python3
"""
Run MAD-path (nonequil DDS) and APT-DEO on the sensor network localisation target.

All steps run end-to-end and the result is saved to a single .npz file.

Usage:
    python run_sensor_network.py --output experiments/results/sensor_network.npz
"""

from __future__ import annotations

import argparse
import os
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

from target.sensor_network import simulate_data as simulate_sensor_network
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path
from sampler.mcmc.adaptive_parallel_tempering import adaptive_parallel_tempering

# ── Target configuration (matches sensor_network.ipynb exactly) ───────────────
_SN_ANCHORS = np.array([[0.5, 0.5], [0.68, 0.8], [0.32, 0.8]], dtype=np.float32)
_TRUE_POSITIONS = np.array([
    [0.23, 0.84],
    [0.58, 0.84],
    [0.56, 0.78],
    [0.80, 0.81],
    [0.30, 0.60],
    [0.49, 0.42],
    [0.42, 0.32],
    [0.80, 0.28],
], dtype=np.float32)

# ── Hyperparameters ───────────────────────────────────────────────────────────
# ULA warm-up
N_ULA       = 500
N_STEPS_ULA = 15_000
ULA_STEP    = 3e-5 / (1.0 + 1e-4)   # h = 3e-5 / (beta + 1e-4) at beta=1, matches notebook

# Gradient ascent to modes
MODE_GD_STEPS     = 50
MODE_GD_LR        = 1e-5
MODE_GRAD_CLIP    = 10.0
MODE_MERGE_RADIUS = 0.1
MIN_MODE_COUNT    = 15

# DDS training
T_FINAL           = 6.0
N_TRAIN           = 300
N_ITER_TRAIN      = 700
N_DIFFUSION_STEPS = 100
DDS_HIDDEN        = 128

# MAD-path chain
NUM_STEPS_MAD = 10_000

# APT-DEO
N_TEMPS_APT    = 50
N_ITER_APT     = 10_000
BURNIN_APT     = 2_000
APT_STEP       = 0.0001
NUM_MALA_STEPS = 20
REF_STD_APT    = 0.1
GAMMA_BASE_APT = 0.4


def main(output: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    target, true_positions = simulate_sensor_network(
        seed=0, r_obs=0.25, sigma_obs=0.01, ref_std=0.05,
        anchors=_SN_ANCHORS, true_positions=_TRUE_POSITIONS,
    )
    D = target.dim
    print(f"Target dim D={D}  observed pairs={len(target.obs_pairs)}")

    # ── Step 1: ULA at full temperature ──────────────────────────────────────
    print("\nStep 1: ULA warm-up...")
    t0 = time.time()

    @jit
    def _run_ula(key):
        key, init_key = jr.split(key)
        x0   = jnp.zeros((N_ULA, D))
        keys = jr.split(key, N_STEPS_ULA)

        def step(x, k):
            score = vmap(jax.grad(target.log_prob))(x)
            return x + ULA_STEP * score + jnp.sqrt(2 * ULA_STEP) * jr.normal(k, x.shape), None

        x_final, _ = jax.lax.scan(step, x0, keys)
        return x_final

    al_samples = np.asarray(_run_ula(jr.PRNGKey(0)))
    print(f"  {time.time()-t0:.1f}s  samples={al_samples.shape[0]}")

    # ── Step 2: Gradient ascent to modes ─────────────────────────────────────
    print("\nStep 2: Mode finding via gradient ascent...")
    t0 = time.time()

    @jit
    def _grad_ascent(x0_batch):
        def step(x, _):
            score = jnp.clip(vmap(jax.grad(target.log_prob))(x), -MODE_GRAD_CLIP, MODE_GRAD_CLIP)
            return x + MODE_GD_LR * score, None
        x_final, _ = jax.lax.scan(step, x0_batch, None, length=MODE_GD_STEPS)
        return x_final

    mode_endpoints = np.asarray(_grad_ascent(jnp.asarray(al_samples)))
    mode_logps = np.asarray(vmap(target.log_prob)(jnp.asarray(mode_endpoints, dtype=jnp.float32)))
    finite = np.isfinite(mode_logps) & np.isfinite(mode_endpoints).all(axis=1)
    mode_endpoints = mode_endpoints[finite]
    mode_logps     = mode_logps[finite]

    _centers: list[np.ndarray] = []
    _counts:  list[int]        = []
    for idx in np.argsort(-mode_logps):
        x = mode_endpoints[idx]
        if not _centers:
            _centers.append(x); _counts.append(1); continue
        centers_arr = np.stack(_centers)
        nearest = int(np.argmin(np.linalg.norm(centers_arr - x[None, :], axis=1)))
        if np.linalg.norm(_centers[nearest] - x) <= MODE_MERGE_RADIUS:
            _counts[nearest] += 1
        else:
            _centers.append(x); _counts.append(1)

    mode_centers = np.stack([c for c, n in zip(_centers, _counts) if n >= MIN_MODE_COUNT])
    mode_counts  = [n for n in _counts if n >= MIN_MODE_COUNT]
    K_fit = len(mode_centers)
    print(f"  {time.time()-t0:.1f}s  modes={K_fit}  counts={mode_counts}")
    if K_fit < 2:
        raise RuntimeError(f"Only {K_fit} mode(s) with count >= {MIN_MODE_COUNT}.")

    # ── Step 3: KMeans + diagonal sample covariances ─────────────────────────
    print("\nStep 3: KMeans GMM fit...")
    km = KMeans(n_clusters=K_fit, init=mode_centers, n_init=1, max_iter=300)
    km.fit(al_samples)
    cluster_var_diag: list[np.ndarray] = []
    weights: list[float] = []
    for k in range(K_fit):
        samp_k = al_samples[km.labels_ == k]
        print(f"  Cluster {k}: {samp_k.shape[0]} samples")
        cluster_var_diag.append(np.diag(np.cov(samp_k, rowvar=False)))
        weights.append(samp_k.shape[0] / al_samples.shape[0])
    gmm_means     = np.asarray(km.cluster_centers_)
    gmm_vars_diag = np.stack(cluster_var_diag)
    weights_np    = np.asarray(weights)
    minimum_stds  = np.min(np.sqrt(gmm_vars_diag), axis=0)

    # ── Step 4: Coordinate rescaling ─────────────────────────────────────────
    shift = jnp.asarray(np.mean(mode_endpoints, axis=0))
    scale = jnp.asarray(minimum_stds)

    def scaled_logp(phi):
        return target.log_prob(phi * scale + shift) + jnp.sum(jnp.log(scale))

    means_s   = (gmm_means - np.asarray(shift)[None, :]) / np.asarray(scale)[None, :]
    vars_s    = gmm_vars_diag / (np.asarray(scale)[None, :] ** 2)
    weights_s = jnp.exp(jnp.log(weights_np) - jax.scipy.special.logsumexp(jnp.log(weights_np)))

    # ── Step 5: Train nonequil DDS ────────────────────────────────────────────
    print("\nStep 5: Training nonequil DDS...")
    t0 = time.time()
    ik, tk, sk = jr.split(jr.PRNGKey(1), 3)

    config = NonequilibriumDDSConfig(
        T_FINAL=T_FINAL, D=D, n_particles=N_TRAIN,
        hidden=DDS_HIDDEN, score_parametrization="base", sigma=1.0,
    )
    dds = make_nonequilibrium_dds_sampler(
        scaled_logp, jnp.asarray(means_s), jnp.asarray(vars_s), weights_s, config,
    )
    params0       = dds.init_params(ik)
    loss_and_grad = dds.make_pathwise_fns(n_steps=N_DIFFUSION_STEPS)
    params, metrics = train(
        params0, loss_and_grad, n_iter=N_ITER_TRAIN,
        lr=1e-2, grad_clip=10.0, log_every=50, key=tk,
    )
    print(f"  {time.time()-t0:.1f}s  final_loss={metrics['losses'][-1]:.4f}")

    # ── Step 6: MAD-path ──────────────────────────────────────────────────────
    print("\nStep 6: MAD-path...")
    t0 = time.time()
    init_scaled, _ = dds.generate_samples(params, sk, n_particles=2000, n_steps=100)
    x0_scaled = init_scaled[0]

    mad_scaled, mad_accept = run_nonequil_mad_path(
        log_target=scaled_logp,
        dds_sampler=dds,
        params=params,
        sigma=1.0,
        T_final=T_FINAL,
        initial_state=x0_scaled,
        num_steps=NUM_STEPS_MAD,
        n_diffusion_steps=N_DIFFUSION_STEPS,
        rng_key=sk,
        with_correction=True,
        n_mala_steps=0,
    )
    mad_samples = np.asarray(mad_scaled) * np.asarray(scale) + np.asarray(shift)
    mad_samples = mad_samples[np.isfinite(mad_samples).all(axis=1)]
    print(f"  {time.time()-t0:.1f}s  accept={float(mad_accept):.3f}  n={mad_samples.shape[0]}")

    # ── Step 7: APT-DEO chain 1 (init: all zeros) ────────────────────────────
    print("\nStep 7: APT-DEO chain 1 (init=zeros)...")
    t0 = time.time()
    apt_sched = jnp.power(0.05, jnp.linspace(0, 1, N_TEMPS_APT))
    apt_deo_samples_1, apt_deo_diag_1 = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=jnp.zeros((N_TEMPS_APT, D)),
        initial_inverse_temperature_schedule=apt_sched,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        initial_step_size=APT_STEP,
        num_mala_steps=NUM_MALA_STEPS,
        rng_key=jr.PRNGKey(4),
        use_doe=True,
        discard_burnin=False,
        gamma_temperature_base=GAMMA_BASE_APT,
        gamma_mala_base=GAMMA_BASE_APT,
    )
    apt_deo_samples_1 = np.asarray(apt_deo_samples_1)
    apt_deo_samples_1 = apt_deo_samples_1[np.isfinite(apt_deo_samples_1).all(axis=1)]
    print(
        f"  {time.time()-t0:.1f}s  "
        f"swap={float(apt_deo_diag_1['swap_acceptance_rate']):.3f}  "
        f"n={apt_deo_samples_1.shape[0]}"
    )

    # ── Step 8: APT-DEO chain 2 (init: all ones) ──────────────────────────────
    print("\nStep 8: APT-DEO chain 2 (init=ones)...")
    t0 = time.time()
    apt_deo_samples_2, apt_deo_diag_2 = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=jnp.ones((N_TEMPS_APT, D)),
        initial_inverse_temperature_schedule=apt_sched,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        initial_step_size=APT_STEP,
        num_mala_steps=NUM_MALA_STEPS,
        rng_key=jr.PRNGKey(5),
        use_doe=True,
        discard_burnin=False,
        gamma_temperature_base=GAMMA_BASE_APT,
        gamma_mala_base=GAMMA_BASE_APT,
    )
    apt_deo_samples_2 = np.asarray(apt_deo_samples_2)
    apt_deo_samples_2 = apt_deo_samples_2[np.isfinite(apt_deo_samples_2).all(axis=1)]
    print(
        f"  {time.time()-t0:.1f}s  "
        f"swap={float(apt_deo_diag_2['swap_acceptance_rate']):.3f}  "
        f"n={apt_deo_samples_2.shape[0]}"
    )

    np.savez(
        output,
        mad_samples        = mad_samples,
        apt_deo_samples_1  = apt_deo_samples_1,
        apt_deo_samples_2  = apt_deo_samples_2,
        mad_accept         = float(mad_accept),
        apt_deo_swap_acc_1 = float(apt_deo_diag_1["swap_acceptance_rate"]),
        apt_deo_swap_acc_2 = float(apt_deo_diag_2["swap_acceptance_rate"]),
        burnin_apt         = BURNIN_APT,
        true_positions     = np.asarray(true_positions),
        anchors            = np.asarray(target.anchors),
        obs_pairs          = np.asarray(target.obs_pairs),
        obs_dists          = np.asarray(target.obs_dists),
        unobs_pairs        = np.asarray(target.unobs_pairs),
    )
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor network MAD-path + APT-DEO")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(_SCRIPT_DIR, "results", "sensor_network.npz"),
    )
    args = parser.parse_args()
    main(args.output)
