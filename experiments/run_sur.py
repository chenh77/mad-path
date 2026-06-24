#!/usr/bin/env python3
"""
Run MAD-Path (Nonequil DDS) and APT-DEO on the SUR profile likelihood (Case 2).

Pipeline:
  1. Annealed Langevin (AL) to explore modes
  2. GMM fit via K-means on AL samples + coordinate rescaling
  3. Non-equilibrium DDS training (scaled space)
  4. MAD-Path chain (scaled → unscaled)
  5. APT-DEO chain (same gradient budget as PT baseline)

Seeds match the notebook exactly (CASE=2, target seed=300).

Saves: experiments/results/sur/case2.npz

Usage (from the repo root):
    python experiments/run_sur.py
    python experiments/run_sur.py --output_dir experiments/results/sur/
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
import blackjax

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from target.sur import make_sur_target
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path
from sampler.mcmc.adaptive_parallel_tempering import adaptive_parallel_tempering

# ── Target (CASE 2) ───────────────────────────────────────────────────────────
_SEED_TARGET = 300
_NOMS = [
    [ 0.0462,  1.7366, -0.3410,  2.2211,  4.0955],
    [ 1.3707,  1.8576, -0.1150,  2.0669,  2.3301],
    [ 0.5028,  1.3007, -0.4648,  0.2104, -0.4490],
]
REF_STD = 1.0

# ── Annealed Langevin ─────────────────────────────────────────────────────────
N_AL       = 200
AL_STEP    = 2e-3
N_AL_STEPS = 3000
GRAD_CLIP  = 20.0

# ── GMM fitting ───────────────────────────────────────────────────────────────
MODE_GD_STEPS     = 50
MODE_GD_LR        = 1e-2
MODE_GRAD_CLIP    = 20.0
MODE_MERGE_RADIUS = 1.0
MIN_MODE_COUNT    = 10

# ── Non-equilibrium DDS ───────────────────────────────────────────────────────
SIGMA_NONEQUIL    = 1.0
T_NONEQUIL        = 6.0
N_PARTICLES_TRAIN = 400
N_ITER_TRAIN      = 500
N_DIFFUSION_TRAIN = 20
DDS_LR            = 1e-2

# ── MAD-Path ──────────────────────────────────────────────────────────────────
NUM_STEPS_MAD   = 12_000
N_DIFFUSION_MAD = 100
N_SAMP_DDS      = 2000   # DDS samples used to pick MAD-Path initial state

# ── APT-DEO (equal gradient budget to notebook PT: 5000×10×20 = 1M) ──────────
N_TEMPS_APT    = 10
N_ITER_APT     = 5000
BURNIN_APT     = 2000   # adaptation phase; post-burnin samples = N_ITER - BURNIN
NUM_MALA_STEPS = 20
INIT_STEP_SIZE = 0.02
GAMMA_BASE     = 0.25    # base learning rate for APT-DEO temp and MALA step size adaptation

# ── NUTS ──────────────────────────────────────────────────────────────────────
NUTS_WARMUP = 5_000
NUTS_N_SAMP = 50_000

# ── PRNG keys (matching notebook) ─────────────────────────────────────────────
AL_KEY        = jr.PRNGKey(0)
DDS_KEY       = jr.PRNGKey(9)    # split into init / train / sample keys
MAD_KEY       = jr.PRNGKey(106)
APT_INIT_KEY  = jr.PRNGKey(20)
APT_RUN_KEY   = jr.PRNGKey(21)
APT_INIT_KEY2 = jr.PRNGKey(20)   # warm start 1: near mode 1
APT_RUN_KEY2  = jr.PRNGKey(21)
APT_INIT_KEY3 = jr.PRNGKey(30)   # warm start 2: near mode 1, different seed
APT_RUN_KEY3  = jr.PRNGKey(31)
NUTS_WARMUP_KEY = jr.PRNGKey(50)
NUTS_RUN_KEY    = jr.PRNGKey(51)


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "case2.npz")

    # ── Build target ─────────────────────────────────────────────────────────
    target = make_sur_target(T=10, M=5, seed=_SEED_TARGET, ref_std=REF_STD)
    D = target.dim
    print(f"SUR target: D={D}  T={target.N}  M={target.M}  seed={_SEED_TARGET}")

    # Refine nominal mode locations with gradient ascent
    _grad_fn = jit(jax.grad(target.log_prob))
    def _ascend(init, n=10000, lr=1e-4, clip=0.1):
        x = jnp.array(init, dtype=jnp.float32)
        for _ in range(n):
            x = x + lr * jnp.clip(_grad_fn(x), -clip, clip)
        return x

    modes = np.array([np.asarray(_ascend(m)) for m in _NOMS])
    param_names = list(target._PARAM_NAMES)
    print("Modes:")
    for i, m in enumerate(modes):
        coords = "  ".join(f"{param_names[j]}={m[j]:.3f}" for j in range(D))
        print(f"  Mode {i+1}: {coords}  log_prob={float(target.log_prob(jnp.asarray(m))):.2f}")

    # ── Step 1: Annealed Langevin ─────────────────────────────────────────────
    print(f"\nStep 1: Annealed Langevin ({N_AL} particles, {N_AL_STEPS} steps)...")
    t0 = time.time()

    REF_VAR = REF_STD ** 2
    beta_schedule = jnp.ones(N_AL_STEPS)   # single temperature (beta=1)
    # Mirror notebook: key, init_key = jr.split(jr.PRNGKey(0))
    al_key_remaining, al_key_init = jr.split(AL_KEY)
    x0_al     = REF_STD * jr.normal(al_key_init, (N_AL, D))
    step_keys = jr.split(al_key_remaining, N_AL_STEPS)

    @jit
    def run_al(x0, keys):
        def step(x, inp):
            beta, k = inp
            eff_step  = AL_STEP * REF_VAR / beta
            tgt_score = jnp.clip(vmap(jax.grad(target.log_prob))(x), -GRAD_CLIP, GRAD_CLIP)
            score     = -(1.0 - beta) * x / REF_VAR + beta * tgt_score
            noise     = jr.normal(k, x.shape)
            return x + eff_step * score + jnp.sqrt(2.0 * eff_step) * noise, None

        x_final, _ = jax.lax.scan(step, x0, (beta_schedule, keys))
        return x_final

    al_samples = np.asarray(run_al(x0_al, step_keys))
    has_nan = bool(np.any(np.isnan(al_samples)))
    print(f"  Done in {time.time()-t0:.1f}s  shape={al_samples.shape}  NaN={has_nan}")
    if has_nan:
        al_samples = al_samples[np.isfinite(al_samples).all(axis=1)]
        print(f"  After NaN filter: {al_samples.shape[0]} particles")

    # ── Step 2: GMM fitting + coordinate rescaling ────────────────────────────
    print("\nStep 2: GMM fitting (gradient ascent + K-means)...")
    t0 = time.time()

    @jit
    def ascend_batch(x0):
        def step(x, _):
            score = jnp.clip(vmap(jax.grad(target.log_prob))(x), -MODE_GRAD_CLIP, MODE_GRAD_CLIP)
            return x + MODE_GD_LR * score, None
        x_final, _ = jax.lax.scan(step, x0, None, length=MODE_GD_STEPS)
        return x_final

    al_jnp        = jnp.array(al_samples, dtype=jnp.float32)
    mode_endpoints = np.asarray(ascend_batch(al_jnp))
    mode_logps    = np.asarray(vmap(target.log_prob)(jnp.asarray(mode_endpoints, dtype=jnp.float32)))
    finite        = np.isfinite(mode_logps) & np.isfinite(mode_endpoints).all(axis=1)
    mode_endpoints = mode_endpoints[finite]
    mode_logps    = mode_logps[finite]

    order = np.argsort(-mode_logps)
    proto_centers, proto_counts = [], []
    for idx in order:
        x = mode_endpoints[idx]
        if not proto_centers:
            proto_centers.append(x); proto_counts.append(1); continue
        nearest = int(np.argmin(np.linalg.norm(np.stack(proto_centers) - x[None], axis=1)))
        if np.linalg.norm(proto_centers[nearest] - x) <= MODE_MERGE_RADIUS:
            proto_counts[nearest] += 1
        else:
            proto_centers.append(x); proto_counts.append(1)

    proto_centers = [c for c, n in zip(proto_centers, proto_counts) if n >= MIN_MODE_COUNT]
    K_FIT = len(proto_centers)
    print(f"  Proto-modes: {K_FIT}")

    km = KMeans(n_clusters=K_FIT, init=np.stack(proto_centers[:K_FIT]), n_init=1)
    km.fit(al_samples)
    gmm_means = np.asarray(km.cluster_centers_)   # (K, D)

    gmm_vars_diag, weights_np = [], []
    for k in range(K_FIT):
        members = al_samples[km.labels_ == k]
        gmm_vars_diag.append(np.diag(np.cov(members, rowvar=False)))
        weights_np.append(members.shape[0] / len(al_samples))
    gmm_vars_diag = np.stack(gmm_vars_diag)   # (K, D)
    weights_np    = np.array(weights_np)

    _shift       = jnp.mean(al_jnp, axis=0)                                      # (D,)
    _coord_scale = jnp.min(jnp.sqrt(jnp.array(gmm_vars_diag)), axis=0)           # (D,)
    _log_jac     = float(jnp.sum(jnp.log(_coord_scale)))

    _means_s = (gmm_means - np.asarray(_shift)) / np.asarray(_coord_scale)       # (K, D)
    _vars_s  = gmm_vars_diag / np.asarray(_coord_scale) ** 2                     # (K, D)

    def log_prob_scaled(phi):
        return target.log_prob(phi * _coord_scale + _shift) + _log_jac

    print(f"  shift       = {np.round(np.asarray(_shift), 4)}")
    print(f"  coord_scale = {np.round(np.asarray(_coord_scale), 4)}")
    print(f"  scaled abs-max per dim = {np.round(np.max(np.abs(_means_s), axis=0), 2)}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 3: Non-equilibrium DDS training ──────────────────────────────────
    print(f"\nStep 3: Nonequil DDS training "
          f"({N_PARTICLES_TRAIN} particles × {N_ITER_TRAIN} iters, T={T_NONEQUIL})...")
    t0 = time.time()

    sigma_vec = jnp.full(D, SIGMA_NONEQUIL, dtype=jnp.float32)
    nonequil_config = NonequilibriumDDSConfig(
        T_FINAL=T_NONEQUIL, D=D, n_particles=N_PARTICLES_TRAIN,
        hidden=64, score_parametrization="base",
        detach_target_score=True, sigma=sigma_vec,
    )
    nonequil_dds = make_nonequilibrium_dds_sampler(
        log_prob_scaled, _means_s, _vars_s, weights_np, nonequil_config,
    )

    ik_ne, tk_ne, sk_ne = jr.split(DDS_KEY, 3)
    params_nonequil, metrics_nonequil = train(
        nonequil_dds.init_params(ik_ne),
        nonequil_dds.make_pathwise_fns(n_steps=N_DIFFUSION_TRAIN),
        n_iter=N_ITER_TRAIN, lr=DDS_LR, grad_clip=10.0, log_every=20, key=tk_ne,
    )
    print(f"  Final loss: {metrics_nonequil['losses'][-1]:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # Generate DDS samples in scaled space to pick MAD-Path initial state
    thetas_s, _ = nonequil_dds.generate_samples(
        params_nonequil, sk_ne, n_particles=N_SAMP_DDS, n_steps=N_DIFFUSION_TRAIN,
    )
    init_mad_s = jnp.array(np.asarray(thetas_s)[0])

    # ── Step 4: MAD-Path chain ────────────────────────────────────────────────
    print(f"\nStep 4: MAD-Path ({NUM_STEPS_MAD} steps, N={N_DIFFUSION_MAD}, T={T_NONEQUIL})...")
    t0 = time.time()

    chain_mad_s, acc_mad = run_nonequil_mad_path(
        log_target=log_prob_scaled,
        dds_sampler=nonequil_dds,
        params=params_nonequil,
        sigma=sigma_vec,
        T_final=T_NONEQUIL,
        initial_state=init_mad_s,
        num_steps=NUM_STEPS_MAD,
        n_diffusion_steps=N_DIFFUSION_MAD,
        n_mala_steps=0,
        mala_step_size=1e-1,
        rng_key=MAD_KEY,
    )
    chain_mad = np.asarray(chain_mad_s) * np.asarray(_coord_scale)[None] + np.asarray(_shift)[None]
    print(f"  Acceptance rate: {acc_mad:.3f}  samples: {chain_mad.shape}  "
          f"({time.time()-t0:.1f}s)")

    # ── Step 5: APT-DEO chain ─────────────────────────────────────────────────
    print(f"\nStep 5: APT-DEO "
          f"({N_ITER_APT} iters, burnin={BURNIN_APT}, "
          f"K={N_TEMPS_APT}, MALA={NUM_MALA_STEPS} steps/iter)...")
    t0 = time.time()

    geo_schedule = list(np.power(0.05, np.linspace(0, 1, N_TEMPS_APT)))
    # Cold start: fixed remote position far from all modes (~15 units away in L2)
    _remote = jnp.full(D, -5.0, dtype=jnp.float32)
    apt_init_states = _remote + 0.1 * jr.normal(APT_INIT_KEY, (N_TEMPS_APT, D))

    chain_apt, apt_diag = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=apt_init_states,
        initial_inverse_temperature_schedule=geo_schedule,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        discard_burnin=False,   # keep full chain; plot_sur slices at BURNIN_APT
        thinning=1,
        initial_step_size=INIT_STEP_SIZE,
        num_mala_steps=NUM_MALA_STEPS,
        rng_key=APT_RUN_KEY,
        use_doe=True,
        gamma_temperature_base=GAMMA_BASE,   # more aggressive temp adaptation for cold start
        gamma_mala_base=GAMMA_BASE,          # more aggressive MALA adaptation for cold start
    )
    chain_apt = np.asarray(chain_apt)
    swap_rate = float(apt_diag["swap_acceptance_rate"])
    mala_rates = np.asarray(apt_diag["mala_acceptance_rate_by_temperature"])
    final_betas = np.asarray(apt_diag["final_inverse_temperatures"])
    print(f"  Cold MALA accept: {mala_rates[0]:.3f}  "
          f"swap rate: {swap_rate:.3f}  "
          f"samples: {chain_apt.shape}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Final temps: {np.round(final_betas, 3)}")

    # ── Step 6: APT-DEO chain (warm start from mode 1) ───────────────────────
    print(f"\nStep 6: APT-DEO warm start "
          f"({N_ITER_APT} iters, burnin={BURNIN_APT}, "
          f"K={N_TEMPS_APT}, MALA={NUM_MALA_STEPS} steps/iter)...")
    t0 = time.time()

    # Initialise all temperature chains near mode 1 with small noise
    apt_init_states2 = (
        jnp.array(modes[0], dtype=jnp.float32)
        + 0.1 * jr.normal(APT_INIT_KEY2, (N_TEMPS_APT, D))
    )

    chain_apt2, apt_diag2 = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=apt_init_states2,
        initial_inverse_temperature_schedule=geo_schedule,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        discard_burnin=False,   # keep full chain; plot_sur slices at BURNIN_APT
        thinning=1,
        initial_step_size=INIT_STEP_SIZE,
        num_mala_steps=NUM_MALA_STEPS,
        rng_key=APT_RUN_KEY2,
        use_doe=True,
        gamma_temperature_base=GAMMA_BASE,   # more aggressive temp adaptation for warm start
        gamma_mala_base=GAMMA_BASE,          # more aggressive MALA adaptation for
    )
    chain_apt2 = np.asarray(chain_apt2)
    swap_rate2  = float(apt_diag2["swap_acceptance_rate"])
    mala_rates2 = np.asarray(apt_diag2["mala_acceptance_rate_by_temperature"])
    final_betas2 = np.asarray(apt_diag2["final_inverse_temperatures"])
    print(f"  Cold MALA accept: {mala_rates2[0]:.3f}  "
          f"swap rate: {swap_rate2:.3f}  "
          f"samples: {chain_apt2.shape}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Final temps: {np.round(final_betas2, 3)}")

    # ── Step 7: APT-DEO chain (warm start 2, different seed) ─────────────────
    print(f"\nStep 7: APT-DEO warm start 2 "
          f"({N_ITER_APT} iters, burnin={BURNIN_APT}, "
          f"K={N_TEMPS_APT}, MALA={NUM_MALA_STEPS} steps/iter)...")
    t0 = time.time()

    apt_init_states3 = (
        jnp.array(modes[0], dtype=jnp.float32)
        + 0.1 * jr.normal(APT_INIT_KEY3, (N_TEMPS_APT, D))
    )

    chain_apt3, apt_diag3 = adaptive_parallel_tempering(
        log_density=target.log_prob,
        initial_states=apt_init_states3,
        initial_inverse_temperature_schedule=geo_schedule,
        num_iterations=N_ITER_APT,
        burnin=BURNIN_APT,
        discard_burnin=False,
        thinning=1,
        initial_step_size=INIT_STEP_SIZE,
        num_mala_steps=NUM_MALA_STEPS,
        rng_key=APT_RUN_KEY3,
        use_doe=True,
        gamma_temperature_base=GAMMA_BASE,
        gamma_mala_base=GAMMA_BASE,
    )
    chain_apt3 = np.asarray(chain_apt3)
    swap_rate3  = float(apt_diag3["swap_acceptance_rate"])
    mala_rates3 = np.asarray(apt_diag3["mala_acceptance_rate_by_temperature"])
    final_betas3 = np.asarray(apt_diag3["final_inverse_temperatures"])
    print(f"  Cold MALA accept: {mala_rates3[0]:.3f}  "
          f"swap rate: {swap_rate3:.3f}  "
          f"samples: {chain_apt3.shape}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  Final temps: {np.round(final_betas3, 3)}")

    # ── Step 8: NUTS reference chain ──────────────────────────────────────────
    print(f"\nStep 8: NUTS reference chain "
          f"(warmup={NUTS_WARMUP}, n_samp={NUTS_N_SAMP})...")
    t0 = time.time()

    nuts_init = jnp.array(modes[0], dtype=jnp.float32)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        target.log_prob,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=0.80,
        progress_bar=False,
    )
    (nuts_state, nuts_params), _ = warmup.run(
        NUTS_WARMUP_KEY, nuts_init, num_steps=NUTS_WARMUP,
    )
    print(f"  step_size = {float(nuts_params['step_size']):.4f}")
    print(f"  inv_mass diag = {jnp.round(nuts_params['inverse_mass_matrix'], 3)}")

    nuts_kernel = blackjax.nuts(target.log_prob, **nuts_params)

    @jit
    def _nuts_step(state, key):
        new_state, info = nuts_kernel.step(key, state)
        return new_state, (new_state.position, info.acceptance_rate)

    nuts_keys = jr.split(NUTS_RUN_KEY, NUTS_N_SAMP)
    _, (nuts_samples_raw, nuts_acc_raw) = jax.lax.scan(
        _nuts_step, nuts_state, nuts_keys,
    )
    chain_nuts = np.asarray(nuts_samples_raw)
    nuts_acc   = float(jnp.mean(nuts_acc_raw))
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"n={NUTS_N_SAMP}  mean_acceptance={nuts_acc:.3f}  shape={chain_nuts.shape}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        out_path,
        chain_mad=chain_mad,
        chain_apt=chain_apt,
        chain_apt2=chain_apt2,
        chain_apt3=chain_apt3,
        chain_nuts=chain_nuts,
        modes=modes,
        param_names=np.array(param_names),
        acc_mad=acc_mad,
        apt_burnin=BURNIN_APT,
        apt_swap_rate=swap_rate,
        apt_mala_rates=mala_rates,
        apt_final_betas=final_betas,
        apt2_swap_rate=swap_rate2,
        apt2_mala_rates=mala_rates2,
        apt2_final_betas=final_betas2,
        apt3_swap_rate=swap_rate3,
        apt3_mala_rates=mala_rates3,
        apt3_final_betas=final_betas3,
        nuts_acc=nuts_acc,
        dds_losses=np.array(metrics_nonequil["losses"]),
        D=D,
    )
    print(f"\nSaved → {out_path}")
    print(f"  chain_mad  : {chain_mad.shape}")
    print(f"  chain_apt  : {chain_apt.shape}")
    print(f"  chain_apt2 : {chain_apt2.shape}")
    print(f"  chain_apt3 : {chain_apt3.shape}")
    print(f"  chain_nuts : {chain_nuts.shape}")


if __name__ == "__main__":
    _default_out = os.path.join(_SCRIPT_DIR, "results", "sur")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=_default_out)
    args = parser.parse_args()
    main(args.output_dir)
