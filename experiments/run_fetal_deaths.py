#!/usr/bin/env python3
"""
T × N sensitivity sweep for MAD-path on the fetal-deaths target.

Two-phase workflow (matches run_bayesian_gmm.py):
  --mode prepare  Run AL, fit GMM, train DDS. Save shared state to --shared_file.
  --mode run      Load shared state, run the T×N sweep for --seed, save seed_NNNN.npz.

AL, GMM fitting, and DDS training are deterministic (no seed) and shared
across all runs; the seed only controls the MAD-path chain RNG.

Metrics per (T, N):
  accept_rate       — MH acceptance rate
  mode1_weight      — fraction with gamma > 0.5  (true ≈ 0.979)
  crossings_per_1k  — inter-mode crossings per 1000 post-burnin steps
  ess_mode          — ESS of the binary mode-indicator sequence
  ess_per_1k_grads  — ess_mode / total_grad_evals × 1000

Mode assignment: theta[0] = logit(gamma) > 0  ↔  gamma > 0.5  →  Mode 1.

Output: one npz per seed with 2-D arrays of shape (len(T_VALS), len(N_VALS)).

Usage (from the repo root):
    python experiments/run_fetal_deaths.py --mode prepare
    python experiments/run_fetal_deaths.py --mode run --seed 0
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

from target.fetal_deaths import make_fetal_deaths_target
from sampler.vi.dds_nonequilibrium import NonequilibriumDDSConfig, make_nonequilibrium_dds_sampler
from sampler.vi.utils import train
from sampler.mcmc.mad_path import run_nonequil_mad_path

# ── Fixed PRNG keys for shared (deterministic) phases ─────────────────────────
_AL_KEY    = jr.PRNGKey(0)
_TRAIN_KEY = jr.PRNGKey(9)

# ── Annealed Langevin (prepare) ────────────────────────────────────────────────
N_AL      = 500
N_STEPS   = 1000
GRAD_CLIP = 20.0
AL_STEP   = 5e-4
REF_STD   = 1.5
_REF_VAR  = REF_STD ** 2
_BETAS    = jnp.ones(1, dtype=jnp.float32)   # single β=1 (full target)

# ── Mode finding (prepare) ─────────────────────────────────────────────────────
MODE_GD_STEPS     = 20
MODE_GD_LR        = 1e-2
MODE_GRAD_CLIP    = 10.0
MODE_MERGE_RADIUS = 1.0
MIN_MODE_COUNT    = 10

# ── DDS training (prepare) ─────────────────────────────────────────────────────
T_TRAIN     = 7.0
N_TRAIN     = 100
N_PARTICLES = 250
N_ITER      = 500
DDS_HIDDEN  = 64
DDS_LR      = 1e-4
SIGMA       = 1.0

# ── T × N sweep grid (run) ─────────────────────────────────────────────────────
T_VALS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]   # sampling horizons (7.0 = T_TRAIN ★)
N_VALS = [10, 20, 30, 40, 50, 60, 70, 100, 150, 300]   # Euler–Maruyama steps per proposal

# ── Chain length (run) ─────────────────────────────────────────────────────────
NUM_STEPS = 10000   # total MCMC steps per chain (matches notebook)
BURNIN    = 5000    # first half discarded

# ── Gradient-evaluation budget (run) ───────────────────────────────────────────
# AL: 500×1000=500k  GD: 500×20=10k  Train: 250×500=125k  → base = 635k
GRADS_BASE = N_AL * N_STEPS + N_AL * MODE_GD_STEPS + N_PARTICLES * N_ITER


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run_annealed_langevin(log_prob, D, key):
    @jit
    def _run(key):
        key, init_key = jr.split(key)
        x0         = REF_STD * jr.normal(init_key, (N_AL, D))
        beta_sched = jnp.repeat(_BETAS, N_STEPS)
        nkeys      = jr.split(key, N_STEPS)

        def step(x, inp):
            beta, k = inp
            tgt_score = jnp.clip(vmap(jax.grad(log_prob))(x), -GRAD_CLIP, GRAD_CLIP)
            score = -(1.0 - beta) * x / _REF_VAR + beta * tgt_score
            return x + AL_STEP * score + jnp.sqrt(2.0 * AL_STEP) * jr.normal(k, x.shape), None

        x_final, _ = jax.lax.scan(step, x0, (beta_sched, nkeys))
        return x_final

    return np.asarray(_run(key))


def _gradient_ascent(log_prob, x0: np.ndarray) -> np.ndarray:
    @jit
    def _run(x):
        def step(x, _):
            score = jnp.clip(vmap(jax.grad(log_prob))(x), -MODE_GRAD_CLIP, MODE_GRAD_CLIP)
            return x + MODE_GD_LR * score, None
        x_final, _ = jax.lax.scan(step, x, None, length=MODE_GD_STEPS)
        return x_final
    return np.asarray(_run(jnp.asarray(x0)))


def _ess_iact(x: np.ndarray) -> float:
    n   = len(x)
    x   = x - x.mean()
    var = float(np.var(x))
    if var < 1e-30:
        return 1.0
    fx   = np.fft.rfft(x, n=2 * n)
    acf  = np.fft.irfft(np.abs(fx) ** 2)[:n].real / (n * var)
    tail = acf[1:]
    tail = tail[:2 * (len(tail) // 2)]
    n_pairs = len(tail) // 2
    gammas  = tail.reshape(n_pairs, 2).sum(axis=1)
    stop    = int(np.argmax(gammas <= 0)) if np.any(gammas <= 0) else n_pairs
    if stop == 0:
        return float(n)
    iact = max(1.0, 1.0 + 2.0 * float(gammas[:stop].sum()))
    return float(n / iact)


def _mode_metrics(samples_unc: np.ndarray, total_grads: int) -> dict:
    """theta[0] = logit(gamma) > 0  ↔  Mode 1 (~97.9%)."""
    in_mode1         = (samples_unc[:, 0] > 0.0).astype(float)
    mode1_weight     = float(in_mode1.mean())
    crossings        = float(np.sum(np.abs(np.diff(in_mode1))))
    crossings_per_1k = crossings / len(samples_unc) * 1000.0
    ess_mode         = _ess_iact(in_mode1)
    ess_per_1k_grads = ess_mode / total_grads * 1000.0
    return dict(
        mode1_weight=mode1_weight,
        crossings_per_1k=crossings_per_1k,
        ess_mode=ess_mode,
        ess_per_1k_grads=ess_per_1k_grads,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — prepare (run once, fully deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

def main_prepare(shared_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(shared_file)), exist_ok=True)

    target = make_fetal_deaths_target(ref_std=REF_STD)
    D      = target.dim
    print(f"Fetal deaths  D={D}  T_TRAIN={T_TRAIN}  N_TRAIN={N_TRAIN}")

    # ── Annealed Langevin ─────────────────────────────────────────────────────
    print("\nStep 1: Annealed Langevin...")
    t0 = time.time()
    al_samples = _run_annealed_langevin(target.log_prob, D, _AL_KEY)
    print(f"  {time.time()-t0:.1f}s  shape={al_samples.shape}  "
          f"NaN={bool(np.any(np.isnan(al_samples)))}")

    # ── Mode finding ──────────────────────────────────────────────────────────
    print("Step 2: Mode finding...")
    t0 = time.time()
    mode_endpoints = _gradient_ascent(target.log_prob, al_samples)
    mode_logps     = np.asarray(
        vmap(target.log_prob)(jnp.asarray(mode_endpoints, dtype=jnp.float32)))
    finite = np.isfinite(mode_logps) & np.all(np.isfinite(mode_endpoints), axis=1)
    mode_endpoints = mode_endpoints[finite]
    mode_logps     = mode_logps[finite]

    order = np.argsort(-mode_logps)
    mode_centers, mode_counts = [], []
    for idx in order:
        x = mode_endpoints[idx]
        if not mode_centers:
            mode_centers.append(x); mode_counts.append(1); continue
        nearest = int(np.argmin(
            np.linalg.norm(np.stack(mode_centers) - x[None, :], axis=1)))
        if np.linalg.norm(mode_centers[nearest] - x) <= MODE_MERGE_RADIUS:
            mode_counts[nearest] += 1
        else:
            mode_centers.append(x); mode_counts.append(1)
    mode_centers = [c for c, n in zip(mode_centers, mode_counts)
                    if n >= MIN_MODE_COUNT]

    K_fit = max(len(mode_centers), 1)
    km    = KMeans(n_clusters=K_fit,
                   init=np.stack(mode_centers[:K_fit]) if mode_centers else "k-means++",
                   n_init=1)
    km.fit(al_samples)

    gmm_means, gmm_vars_diag, weights_np = [], [], []
    for k in range(K_fit):
        members = al_samples[km.labels_ == k]
        gmm_means.append(km.cluster_centers_[k])
        gmm_vars_diag.append(np.diag(np.cov(members, rowvar=False)))
        weights_np.append(members.shape[0] / len(al_samples))
    gmm_means     = np.stack(gmm_means)
    gmm_vars_diag = np.stack(gmm_vars_diag)
    weights_np    = np.array(weights_np)
    print(f"  {time.time()-t0:.1f}s  K_fit={K_fit}  "
          f"weights={[f'{w:.3f}' for w in weights_np]}")
    for k in range(K_fit):
        g, mu, th, nu = target.constrained(jnp.array(gmm_means[k], jnp.float32))
        print(f"  mode {k}: gamma={float(g):.3f}  mu={float(mu):.3f}  "
              f"theta={float(th):.3f}  nu={float(nu):.3f}")

    # ── Coordinate rescaling ──────────────────────────────────────────────────
    shift       = np.asarray(jnp.asarray(al_samples.mean(axis=0), dtype=jnp.float32))
    coord_scale = np.sqrt(gmm_vars_diag).min(axis=0).astype(np.float32)
    log_jac     = float(np.sum(np.log(coord_scale)))
    means_s     = (gmm_means - shift[None, :]) / coord_scale[None, :]
    vars_s      = gmm_vars_diag / coord_scale[None, :] ** 2
    print(f"  shift={np.round(shift, 4)}  coord_scale={np.round(coord_scale, 4)}")

    def log_prob_scaled(phi):
        return target.log_prob(
            phi * jnp.asarray(coord_scale) + jnp.asarray(shift)) + log_jac

    sigma_vec = jnp.full(D, SIGMA, dtype=jnp.float32)

    # ── Nonequil DDS training ─────────────────────────────────────────────────
    print(f"\nStep 3: Training nonequil DDS  "
          f"(T={T_TRAIN}, N={N_TRAIN}, n_iter={N_ITER}, lr={DDS_LR})...")
    t0 = time.time()
    config = NonequilibriumDDSConfig(
        T_FINAL=T_TRAIN, D=D,
        n_particles=N_PARTICLES,
        hidden=DDS_HIDDEN,
        score_parametrization="base",
        detach_target_score=True,
        sigma=sigma_vec,
    )
    dds = make_nonequilibrium_dds_sampler(
        log_prob_scaled,
        jnp.asarray(means_s, dtype=jnp.float32),
        jnp.asarray(vars_s,  dtype=jnp.float32),
        jnp.asarray(weights_np / weights_np.sum(), dtype=jnp.float32),
        config,
    )
    ik_ne, tk_ne, sk_ne = jr.split(_TRAIN_KEY, 3)
    params, metrics = train(
        dds.init_params(ik_ne),
        dds.make_pathwise_fns(n_steps=N_TRAIN),
        n_iter=N_ITER, lr=DDS_LR, grad_clip=10.0,
        log_every=100, key=tk_ne,
    )
    final_loss = float(metrics["losses"][-1]) if metrics["losses"] else float("nan")
    print(f"  {time.time()-t0:.1f}s  final_loss={final_loss:.4f}")

    # Initial state (matches notebook: thetas_nonequil_s[0] using sk_ne)
    init_s, _ = dds.generate_samples(params, sk_ne, n_particles=2000, n_steps=N_TRAIN)
    x0_s = np.asarray(init_s[0])

    # ── Save shared state ──────────────────────────────────────────────────────
    shared = dict(
        shift=shift,
        coord_scale=coord_scale,
        log_jac=log_jac,
        means_s=means_s,
        vars_s=vars_s,
        weights_np=weights_np / weights_np.sum(),
        params=params,
        x0_s=x0_s,
        K_fit=K_fit,
        T_train=T_TRAIN,
        N_train=N_TRAIN,
        final_loss=final_loss,
    )
    with open(shared_file, "wb") as f:
        pickle.dump(shared, f)
    print(f"\nShared state saved → {shared_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — run (T × N sweep, one per seed)
# ═══════════════════════════════════════════════════════════════════════════════

def main_run(seed: int, shared_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seed_{seed:04d}.npz")

    if os.path.exists(out_path):
        print(f"Already exists: {out_path}  →  skipping.")
        return

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

    target = make_fetal_deaths_target(ref_std=REF_STD)
    D      = target.dim

    def log_prob_scaled(phi):
        return target.log_prob(phi * coord_scale + shift) + log_jac

    sigma_vec = jnp.full(D, SIGMA, dtype=jnp.float32)

    # Rebuild DDS sampler (no training — params already loaded)
    config = NonequilibriumDDSConfig(
        T_FINAL=T_TRAIN, D=D,
        n_particles=N_PARTICLES,
        hidden=DDS_HIDDEN,
        score_parametrization="base",
        detach_target_score=True,
        sigma=sigma_vec,
    )
    dds = make_nonequilibrium_dds_sampler(
        log_prob_scaled, means_s, vars_s, weights_s, config)

    print(f"seed={seed}  D={D}  T_TRAIN={T_TRAIN}  N_TRAIN={N_TRAIN}  "
          f"final_loss={saved.get('final_loss', float('nan')):.4f}")
    print(f"T_VALS={T_VALS}  N_VALS={N_VALS}")

    # ── T × N sweep ───────────────────────────────────────────────────────────
    n_T, n_N = len(T_VALS), len(N_VALS)
    accept_grid    = np.zeros((n_T, n_N))
    mode1_grid     = np.zeros((n_T, n_N))
    crossings_grid = np.zeros((n_T, n_N))
    ess_mode_grid  = np.zeros((n_T, n_N))
    ess_grads_grid = np.zeros((n_T, n_N))

    print(f"\nMAD-path sweep ({n_T} T × {n_N} N, "
          f"num_steps={NUM_STEPS}, burnin={BURNIN})...")
    exp_key = jr.PRNGKey(seed)
    for i, T in enumerate(T_VALS):
        for j, N in enumerate(N_VALS):
            print(f"  T={T:5.1f}  N={N:4d} ...", end="", flush=True)
            t0 = time.time()
            exp_key, chain_key = jr.split(exp_key)
            chain_s, accept = run_nonequil_mad_path(
                log_target=log_prob_scaled,
                dds_sampler=dds,
                params=params,
                sigma=sigma_vec,
                T_final=T,
                initial_state=x0_s,
                num_steps=NUM_STEPS,
                n_diffusion_steps=N,
                rng_key=chain_key,
                with_correction=True,
                burnin=BURNIN,
            )
            chain_unc = (np.asarray(chain_s) * np.asarray(coord_scale)[None, :]
                         + np.asarray(shift)[None, :])

            total_grads = GRADS_BASE + NUM_STEPS * N
            m = _mode_metrics(chain_unc, total_grads)

            accept_grid[i, j]    = accept
            mode1_grid[i, j]     = m["mode1_weight"]
            crossings_grid[i, j] = m["crossings_per_1k"]
            ess_mode_grid[i, j]  = m["ess_mode"]
            ess_grads_grid[i, j] = m["ess_per_1k_grads"]

            print(f"  {time.time()-t0:.1f}s  "
                  f"accept={accept:.3f}  "
                  f"cross/1k={m['crossings_per_1k']:.2f}  "
                  f"ess={m['ess_mode']:.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        seed=seed,
        T_train=T_TRAIN,
        N_train=N_TRAIN,
        T_vals=np.array(T_VALS),
        N_vals=np.array(N_VALS),
        accept_rate=accept_grid,
        mode1_weight=mode1_grid,
        crossings_per_1k=crossings_grid,
        ess_mode=ess_mode_grid,
        ess_per_1k_grads=ess_grads_grid,
        grads_base=GRADS_BASE,
        num_steps_mad=NUM_STEPS,
    )
    print(f"\nSaved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _RESULTS = os.path.join(_SCRIPT_DIR, "results")
    parser   = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["prepare", "run"], required=True)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--shared_file", type=str,
                        default=os.path.join(_RESULTS, "fetal_deaths_shared.pkl"))
    parser.add_argument("--output_dir",  type=str,
                        default=os.path.join(_RESULTS, "fetal_deaths"))
    args = parser.parse_args()

    if args.mode == "prepare":
        main_prepare(args.shared_file)
    else:
        main_run(args.seed, args.shared_file, args.output_dir)
