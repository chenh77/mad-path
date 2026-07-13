#!/usr/bin/env python3
"""
Run all samplers on the D=20 bimodal GMM and save results to
experiments/results/gmm_d20_comparison.npz.

Pipeline (mirrors dds.ipynb):
  1. Annealed Langevin  →  covering samples
  2. k-means GMM fit    →  reference process for nonequil DDS
  3. Train nonequil DDS →  score network
  4. MAD-Path MCMC      →  chain from score-corrected kernel
  5. APT (no DEO)
  6. APT + DEO
  7. Tempered Transition
  8. AIS
  9. SMC

Usage:
    cd experiments/
    python run_samplers.py [--seed 0] [--output results/gmm_d20_comparison.npz]
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

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path
from sampler.vi.dds_sampling import run_nonequil_dds
from sampler.mcmc.adaptive_parallel_tempering import adaptive_parallel_tempering
from sampler.mcmc.tempered_transition import tempered_transition
from sampler.importance_sampling.ais import adaptive_annealed_importance_sampling
from sampler.importance_sampling.smc import adaptive_sequential_monte_carlo

# ═══════════════════════════════════════════════════════════════════════════════
# Target: isotropic Gaussian mixture, D=20
# ═══════════════════════════════════════════════════════════════════════════════
D = 20
K = 2
TGT_MEANS      = 1.0 * jnp.array([jnp.ones(D), -jnp.ones(D)])
TGT_STD_SCALAR = jnp.array([jnp.sqrt(0.5), jnp.sqrt(0.2)])
TGT_LOG_VAR    = jnp.log(TGT_STD_SCALAR ** 2)
TGT_LOG_W      = jnp.log(jnp.ones(K) / K)
REF_STD        = 1.1

def log_gmm_iso(x, means, log_var, log_weights):
    var  = jnp.exp(log_var)
    diff = x[None, :] - means
    lp   = (-0.5 * jnp.sum(diff ** 2, axis=-1) / var
            - 0.5 * D * jnp.log(var)
            - 0.5 * D * jnp.log(2.0 * jnp.pi))
    return jax.scipy.special.logsumexp(log_weights + lp)

def log_target_density(x):
    return log_gmm_iso(x, TGT_MEANS, TGT_LOG_VAR, TGT_LOG_W)

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Annealed Langevin (covering samples for GMM fit)
# ═══════════════════════════════════════════════════════════════════════════════
def run_annealed_langevin(key, n_particles=1000, n_temps=20, n_steps=20, step_size=0.1):
    betas = jnp.power(0.9, jnp.arange(n_temps))[::-1]

    @jit
    def _run(key):
        key, ik = jr.split(key)
        x0 = REF_STD * jr.normal(ik, (n_particles, D))
        beta_schedule = jnp.repeat(betas, n_steps)
        keys = jr.split(key, n_temps * n_steps)
        ref_var = REF_STD ** 2

        def step(x, inp):
            beta, k = inp
            score = (-(1 - beta) * x / ref_var
                     + beta * vmap(jax.grad(log_target_density))(x))
            return x + step_size * score + jnp.sqrt(2 * step_size) * jr.normal(k, x.shape), None

        x_final, _ = jax.lax.scan(step, x0, (beta_schedule, keys))
        return x_final

    return np.asarray(_run(key))

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Fit isotropic GMM via k-means + moment matching
# ═══════════════════════════════════════════════════════════════════════════════
def fit_gmm_from_samples(samples, K_fit=2, n_iter=50):
    n, d = samples.shape
    s = np.asarray(samples)
    rng = np.random.default_rng(42)

    # k-means++ init
    means = [s[rng.integers(n)]]
    for _ in range(1, K_fit):
        stack = np.stack(means)
        dists = np.min(np.sum((s[:, None, :] - stack[None, :, :]) ** 2, axis=-1), axis=-1)
        means.append(s[np.argmax(dists)])
    means = np.stack(means)

    for _ in range(n_iter):
        dists   = np.sum((s[:, None, :] - means[None, :, :]) ** 2, axis=-1)
        assigns = np.argmin(dists, axis=-1)
        for k in range(K_fit):
            mask = assigns == k
            if np.sum(mask) > 0:
                means[k] = np.mean(s[mask], axis=0)

    dists   = np.sum((s[:, None, :] - means[None, :, :]) ** 2, axis=-1)
    assigns = np.argmin(dists, axis=-1)
    log_vars, weights = [], []
    for k in range(K_fit):
        mask  = assigns == k
        count = int(np.sum(mask))
        weights.append(count / n)
        var_k = float(np.sum((s[mask] - means[k]) ** 2) / (count * d))
        log_vars.append(np.log(max(var_k, 1e-6)))

    return {
        "means":    jnp.array(means),
        "log_vars": jnp.array(log_vars),
        "logits":   jnp.log(jnp.clip(jnp.array(weights), 1e-8)) * 0.5,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main(seed: int = 0, output: str | None = None):
    if output is None:
        output = os.path.join(_SCRIPT_DIR, "results", "gmm_d20_comparison.npz")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    key = jr.PRNGKey(seed)
    results = {}

    # ── 1. Annealed Langevin ─────────────────────────────────────────────────
    print("Step 1: Annealed Langevin...")
    t0 = time.time()
    key, sk = jr.split(key)
    al_samples = run_annealed_langevin(sk)
    print(f"  done in {time.time()-t0:.1f}s  shape={al_samples.shape}")

    # ── 2. Fit GMM ────────────────────────────────────────────────────────────
    print("Step 2: Fitting GMM...")
    gmm_p = fit_gmm_from_samples(al_samples, K_fit=2)
    lw_fit = gmm_p["logits"] - jax.scipy.special.logsumexp(gmm_p["logits"])
    for k in range(2):
        v1 = jnp.ones(D) / jnp.sqrt(D)
        print(f"  comp {k}: wt={float(jnp.exp(lw_fit[k])):.3f}  "
              f"var={float(jnp.exp(gmm_p['log_vars'][k])):.4f}  "
              f"mean·v1={float(gmm_p['means'][k] @ v1):.3f}")

    # ── 3. Train Nonequil DDS ─────────────────────────────────────────────────
    print("Step 3: Training nonequilibrium DDS...")
    nonequil_config = NonequilibriumDDSConfig(
        T_FINAL=5.0, D=D, n_particles=512, hidden=64,
        score_parametrization="base", sigma=1.0,
    )
    nonequil_means   = gmm_p["means"]
    nonequil_covs    = jnp.exp(gmm_p["log_vars"])
    nonequil_weights = jnp.exp(lw_fit)

    nonequil_dds = make_nonequilibrium_dds_sampler(
        log_target_density, nonequil_means, nonequil_covs, nonequil_weights,
        nonequil_config,
    )
    key, ik, tk = jr.split(key, 3)
    params0       = nonequil_dds.init_params(ik)
    loss_and_grad = nonequil_dds.make_pathwise_fns(n_steps=50)
    t0 = time.time()
    nonequil_params, metrics = train(
        params0, loss_and_grad, n_iter=200,
        lr=1e-3, grad_clip=10.0, log_every=20, key=tk,
    )
    print(f"  done in {time.time()-t0:.1f}s  final loss={metrics['losses'][-1]:.4f}")

    # ── 4. MAD-Path ───────────────────────────────────────────────────────────
    print("Step 4: MAD-Path MCMC...")
    t0 = time.time()
    key, sk = jr.split(key)
    mad_samples, mad_accept = run_nonequil_mad_path(
        log_target_density,
        nonequil_dds,
        nonequil_params,
        sigma=nonequil_config.sigma,
        T_final=nonequil_config.T_FINAL,
        initial_state=jnp.zeros(D),
        num_steps=10000,
        n_diffusion_steps=50,
        rng_key=sk,
        burnin=0,
    )
    mad_samples = np.asarray(mad_samples)
    print(f"  done in {time.time()-t0:.1f}s  accept={mad_accept:.3f}  "
          f"n_samples={mad_samples.shape[0]}")
    results["mad_path_samples"] = mad_samples

    # ── 4b. DDS (nonequil) ───────────────────────────────────────────────────
    print("Step 4b: DDS (nonequilibrium)...")
    t0 = time.time()
    key, sk = jr.split(key)
    dds_samples, dds_log_w = run_nonequil_dds(
        nonequil_dds,
        nonequil_params,
        sigma=nonequil_config.sigma,
        n_samples=10000,
        n_diffusion_steps=50,
        T_final=nonequil_config.T_FINAL,
        rng_key=sk,
        log_target=log_target_density,
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
    n_temps    = 20
    apt_sched  = jnp.power(0.01, jnp.linspace(0, 1, n_temps))
    key, sk = jr.split(key)
    apt_samples, apt_diag = adaptive_parallel_tempering(
        log_density=log_target_density,
        initial_states=jr.normal(sk, (n_temps, D)),
        initial_inverse_temperature_schedule=apt_sched,
        num_iterations=12500,
        burnin=2500,
        initial_step_size=0.01,
        num_mala_steps=10,
        rng_key=jr.PRNGKey(seed + 10),
        use_deo=False,
    )
    apt_samples = np.asarray(apt_samples)
    print(f"  done in {time.time()-t0:.1f}s  swap_accept={float(apt_diag['swap_acceptance_rate']):.3f}  "
          f"n_samples={apt_samples.shape[0]}")
    results["apt_samples"] = apt_samples

    # ── 6. APT + DEO ─────────────────────────────────────────────────────────
    print("Step 6: APT + DEO...")
    t0 = time.time()
    key, sk = jr.split(key)
    apt_sched  = jnp.power(0.3, jnp.linspace(0, 1, n_temps))
    apt_deo_samples, apt_deo_diag = adaptive_parallel_tempering(
        log_density=log_target_density,
        initial_states=jr.normal(sk, (n_temps, D)),
        initial_inverse_temperature_schedule=apt_sched,
        num_iterations=12500,
        burnin=2500,
        initial_step_size=0.3,
        num_mala_steps=10,
        rng_key=jr.PRNGKey(seed + 10),
        use_deo=True,
        # Tightened from the default (-20, 5): at the hot end here MALA
        # acceptance sits pinned near 1.0, so Robbins-Monro step-size
        # adaptation can inflate unboundedly and overflow float32. This
        # bound rate-limits rho updates without changing the returned
        # cold-chain samples.
        rho_bounds=(-20.0, 1.5),
    )
    apt_deo_samples = np.asarray(apt_deo_samples)
    print(f"  done in {time.time()-t0:.1f}s  swap_accept={float(apt_deo_diag['swap_acceptance_rate']):.3f}  "
          f"n_samples={apt_deo_samples.shape[0]}")
    results["apt_deo_samples"] = apt_deo_samples

    # ── 7. Tempered Transition (hand-tuned schedule) ────────────────────────────
    print("Step 7: Tempered Transition (hand-tuned schedule)...")
    t0 = time.time()
    _tt_betas = jnp.geomspace(1.0, 0.02, 15)
    _tt_steps = 0.005 / jnp.sqrt(_tt_betas)
    tt_samples, tt_accept = tempered_transition(
        log_density=log_target_density,
        initial_state=jr.normal(jr.PRNGKey(seed + 30), (D,)),
        inverse_temperature_schedule=_tt_betas,
        num_iterations=12500,
        burnin=2500,
        step_size=_tt_steps,
        num_steps=10,
        transition_type="mala",
        rng_key=jr.PRNGKey(seed + 30),
    )
    tt_samples = np.asarray(tt_samples)
    print(f"  done in {time.time()-t0:.1f}s  accept={float(tt_accept):.3f}  "
          f"n_samples={tt_samples.shape[0]}")
    results["tt_samples"] = tt_samples

    # ── 8. Adaptive AIS ───────────────────────────────────────────────────────
    print("Step 8: Adaptive AIS (Zhou et al. 2016)...")
    t0 = time.time()
    ais_samples, ais_log_w, ais_info = adaptive_annealed_importance_sampling(
        log_density=log_target_density,
        D=D,
        ref_std=REF_STD,
        num_samples=10_000,
        step_size=0.25,
        num_steps=10,
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
        log_density=log_target_density,
        D=D,
        ref_std=REF_STD,
        num_particles=10_000,
        step_size=0.8,
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
