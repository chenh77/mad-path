#!/usr/bin/env python3
"""
Run all samplers on the D=20 skew-normal mixture synthetic target and save
results to experiments/results/synthetic_comparison.npz.

Pipeline (mirrors dds_synthetic.ipynb):
  1-2. Annealed Langevin + mode finding + k-means GMM fit
       (sampler.vi.mode_initialization.initialize_modes_from_langevin) →  reference process
  3. Train nonequil DDS   →  score network
  4. MAD-Path MCMC        →  chain from score-corrected kernel
  5. APT (no DEO)
  6. APT + DEO
  7. Tempered Transition
  8. AIS
  9. SMC

Usage:
    cd experiments/
    python run_samplers_synthetic.py [--seed 0] [--output results/synthetic_comparison.npz]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from target.skew import make_skew_mixture_target
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path
from sampler.vi.dds_sampling import run_nonequil_dds
from sampler.mcmc.adaptive_parallel_tempering import adaptive_parallel_tempering
from sampler.mcmc.tempered_transition import tempered_transition
from sampler.importance_sampling.ais import adaptive_annealed_importance_sampling
from sampler.importance_sampling.smc import adaptive_sequential_monte_carlo
from sampler.vi.mode_initialization import initialize_modes_from_langevin

# ═══════════════════════════════════════════════════════════════════════════════
# Target: Skew-Normal Mixture, D=20, K=13 components
# ═══════════════════════════════════════════════════════════════════════════════
D = 20
REF_STD = 10.0

def make_target():
    return make_skew_mixture_target(
        d=D,
        mu_levels=(5.0, 10.0),
        sigma=1.0,
        alpha=5.0,
        eps_tie=1e-3,
        pair_jitter=0.0,
        seed=0,
        ref_std=REF_STD,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main(seed: int = 0, output: str | None = None):
    if output is None:
        output = os.path.join(_SCRIPT_DIR, "results", "synthetic_comparison.npz")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    target = make_target()
    log_prob = target.log_prob

    key = jr.PRNGKey(seed)
    results = {}

    # save true mode centers and true samples for plotting
    results["target_mus"] = np.asarray(target.mus)
    key, sk = jr.split(key)
    true_samples, _ = target.sample(sk, 5000)
    results["true_samples"] = np.asarray(true_samples)

    # ── 1+2. Annealed Langevin + gradient-ascent mode finding + k-means fit ───
    print("Step 1-2: Annealed Langevin + mode finding + k-means GMM fit...")
    t0 = time.time()
    key, sk = jr.split(key)
    mode_result = initialize_modes_from_langevin(
        target,
        langevin_step_size=0.05,
        beta_schedule=jnp.power(0.01, 1 - jnp.linspace(0, 1, 10)),
        num_particles=1000,
        num_langevin_steps_per_beta=100,
        target_num_modes=13,
        mode_gd_steps=20, mode_gd_step_size=0.1, mode_grad_clip=10.0,
        merge_radius_use_fixed_stepsize=True, merge_radius_multiplier=20.0,
        min_mode_count=1,
        rng_key=sk,
    )
    K_fit = mode_result.modes.shape[0]
    # uniform weights, matching the skew target's true w_k = 1/13 (mirrors the
    # old fit_gmm_from_samples, which ignored per-cluster sample counts)
    gmm_p = {
        "means":  mode_result.modes,
        "vars":   mode_result.covariances_diag,
        "logits": jnp.zeros(K_fit),
    }
    print(f"  Found {K_fit} modes in {time.time()-t0:.1f}s  "
          f"(from {mode_result.langevin_samples.shape[0]} AL samples)")

    # ── 3. Train Nonequil DDS ─────────────────────────────────────────────────
    print("Step 3: Training nonequilibrium DDS...")
    nonequil_config = NonequilibriumDDSConfig(
        T_FINAL=5.0, D=D, n_particles=512, hidden=64,
        score_parametrization="base", sigma=1.0,
    )
    weights = jnp.exp(gmm_p["logits"] - jax.scipy.special.logsumexp(gmm_p["logits"]))
    nonequil_dds = make_nonequilibrium_dds_sampler(
        log_prob, gmm_p["means"], gmm_p["vars"], weights, nonequil_config,
    )
    key, ik, tk, sk = jr.split(key, 4)
    params0 = nonequil_dds.init_params(ik)
    loss_and_grad = nonequil_dds.make_pathwise_fns(n_steps=100)
    t0 = time.time()
    nonequil_params, metrics = train(
        params0, loss_and_grad, n_iter=50,
        lr=1e-4, grad_clip=10.0, log_every=10, key=tk,
    )
    print(f"  done in {time.time()-t0:.1f}s  final loss={metrics['losses'][-1]:.4f}")

    # ── 4. MAD-Path ───────────────────────────────────────────────────────────
    print("Step 4: MAD-Path MCMC...")
    t0 = time.time()
    mad_samples, mad_accept = run_nonequil_mad_path(
        log_target=log_prob,
        dds_sampler=nonequil_dds,
        params=nonequil_params,
        sigma=1.0,
        T_final=5.0,
        initial_state=jnp.zeros(D),
        num_steps=10000,
        n_diffusion_steps=100,
        rng_key=sk,
        burnin=0,
        with_correction=True,
    )
    mad_samples = np.asarray(mad_samples)
    print(f"  done in {time.time()-t0:.1f}s  accept={mad_accept:.3f}  n={mad_samples.shape[0]}")
    results["mad_path_samples"] = mad_samples

    # ── 4b. DDS (nonequil) ───────────────────────────────────────────────────
    print("Step 4b: DDS (nonequilibrium)...")
    t0 = time.time()
    key, sk = jr.split(key)
    dds_samples, dds_log_w = run_nonequil_dds(
        nonequil_dds,
        nonequil_params,
        sigma=1.0,
        n_samples=10000,
        n_diffusion_steps=100,
        T_final=5.0,
        rng_key=sk,
        log_target=log_prob,
    )
    dds_samples = np.asarray(dds_samples)
    dds_log_w   = np.asarray(dds_log_w)
    w_norm = np.exp(dds_log_w - dds_log_w.max())
    ess    = 1.0 / np.sum((w_norm / w_norm.sum()) ** 2)
    print(f"  done in {time.time()-t0:.1f}s  ESS={ess:.1f}/{len(dds_samples)}")
    results["dds_samples"]     = dds_samples
    results["dds_log_weights"] = dds_log_w

    # ── 5. APT (no DEO) ───────────────────────────────────────────────────────
    print("Step 5: APT (no DEO)...")
    t0 = time.time()
    n_temps   = 20
    apt_sched = jnp.power(0.02, jnp.linspace(0, 1, n_temps))
    key, sk = jr.split(key)
    apt_samples, apt_diag = adaptive_parallel_tempering(
        log_density=log_prob,
        initial_states=jr.normal(sk, (n_temps, D)),
        initial_inverse_temperature_schedule=apt_sched,
        num_iterations=10000,
        burnin=2000,
        initial_step_size=0.02,
        num_mala_steps=10,
        rng_key=jr.PRNGKey(seed + 10),
        use_deo=False,
    )
    apt_samples = np.asarray(apt_samples)
    print(f"  done in {time.time()-t0:.1f}s  swap_accept={float(apt_diag['swap_acceptance_rate']):.3f}  n={apt_samples.shape[0]}")
    results["apt_samples"] = apt_samples

    # ── 6. APT + DEO ──────────────────────────────────────────────────────────
    print("Step 6: APT + DEO...")
    t0 = time.time()
    key, sk = jr.split(key)
    apt_deo_samples, apt_deo_diag = adaptive_parallel_tempering(
        log_density=log_prob,
        initial_states=jr.normal(sk, (n_temps, D)),
        initial_inverse_temperature_schedule=jnp.power(0.9, jnp.linspace(0, 1, n_temps)),
        num_iterations=10000,
        burnin=2000,
        initial_step_size=0.5,
        num_mala_steps=10,
        rng_key=jr.PRNGKey(seed + 10),
        use_deo=True,
    )
    apt_deo_samples = np.asarray(apt_deo_samples)
    print(f"  done in {time.time()-t0:.1f}s  swap_accept={float(apt_deo_diag['swap_acceptance_rate']):.3f}  n={apt_deo_samples.shape[0]}")
    results["apt_deo_samples"] = apt_deo_samples

    # ── 7. Tempered Transition (hand-tuned schedule) ────────────────────────────
    print("Step 7: Tempered Transition (hand-tuned schedule)...")
    t0 = time.time()
    _tt_betas = jnp.geomspace(1.0, 0.05, 10)
    _tt_steps = 0.01 / jnp.sqrt(_tt_betas)
    tt_samples, tt_accept = tempered_transition(
        log_density=log_prob,
        initial_state=jr.normal(jr.PRNGKey(seed + 30), (D,)),
        inverse_temperature_schedule=_tt_betas,
        num_iterations=10000,
        burnin=2000,
        step_size=_tt_steps,
        num_steps=10,
        transition_type="mala",
        rng_key=jr.PRNGKey(seed + 31),
    )
    tt_samples = np.asarray(tt_samples)
    print(f"  done in {time.time()-t0:.1f}s  accept={float(tt_accept):.3f}  n={tt_samples.shape[0]}")
    results["tt_samples"] = tt_samples

    # ── 8. Adaptive AIS ───────────────────────────────────────────────────────
    print("Step 8: Adaptive AIS (Zhou et al. 2016)...")
    t0 = time.time()
    ais_samples, ais_log_w, ais_info = adaptive_annealed_importance_sampling(
        log_density=log_prob,
        D=D,
        ref_std=REF_STD,
        num_samples=10_000,
        step_size=0.75,
        num_steps=15,
        target_cess=0.5,
        max_levels=500,
        rng_key=jr.PRNGKey(seed + 40),
        transition_type="mala",
        return_info=True,
    )
    ais_samples = np.asarray(ais_samples)
    ais_log_w   = np.asarray(ais_log_w)
    w_norm = np.exp(ais_log_w - ais_log_w.max())
    ess    = 1.0 / np.sum((w_norm / w_norm.sum()) ** 2)
    print(f"  done in {time.time()-t0:.1f}s  accept={float(ais_info['acceptance_rate']):.3f}  "
          f"levels={int(ais_info['num_levels'])}  ESS={ess:.1f}/{len(ais_samples)}")
    results["ais_samples"]     = ais_samples
    results["ais_log_weights"] = ais_log_w

    # ── 9. Adaptive SMC ───────────────────────────────────────────────────────
    print("Step 9: Adaptive SMC (Zhou et al. 2016)...")
    t0 = time.time()
    smc_samples, smc_log_w, smc_info = adaptive_sequential_monte_carlo(
        log_density=log_prob,
        D=D,
        ref_std=REF_STD,
        num_particles=10_000,
        step_size=0.5,
        num_steps=10,
        target_cess=0.5,
        max_levels=500,
        rng_key=jr.PRNGKey(seed + 50),
        transition_type="mala",
        return_info=True,
    )
    smc_samples = np.asarray(smc_samples)
    smc_log_w   = np.asarray(smc_log_w)
    w_norm = np.exp(smc_log_w - smc_log_w.max())
    ess    = 1.0 / np.sum((w_norm / w_norm.sum()) ** 2)
    print(f"  done in {time.time()-t0:.1f}s  accept={float(smc_info['acceptance_rate']):.3f}  "
          f"levels={int(smc_info['num_levels'])}  ESS={ess:.1f}/{len(smc_samples)}")
    results["smc_samples"]     = smc_samples
    results["smc_log_weights"] = smc_log_w

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(output, **results)
    print(f"\nSaved {output}")
    for k, v in results.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(seed=args.seed, output=args.output)
