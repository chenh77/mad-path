"""Pure reverse-SDE sampler for the nonequilibrium (reverse-OU) DDS.

Runs the controlled backward SDE N independent times, each starting from
x_0 ~ N(0, σ²I), using the reverse-OU base drift augmented by the learned
correction f_θ:

    dY_t = [½Y_t + σ² ∇log p_{T-t}(Y_t) + σ² f_θ(Y_t, t)] dt + σ dW_t,
    Y_0 ~ N(0, σ²I)

where p_s is the OU marginal of the GMM reference at time s = T − t.
There is no forward (noising) leg and no Metropolis–Hastings correction.

Mirrors the structure of mad_path.py (make_nonequil_mh_step / run_nonequil_mad_path)
but drops the forward leg and MH acceptance step.

Public API
----------
make_nonequil_dds_sample_fn(dds_sampler, params, sigma, n_diffusion_steps,
                             T_final, *, with_correction=True)
    -> JIT-compiled  (x0, key) -> (x_T, log_girsanov)

run_nonequil_dds(dds_sampler, params, sigma, n_samples, n_diffusion_steps,
                 T_final, rng_key, *, log_target=None, with_correction=True)
    -> (samples, log_weights)
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

Array = jax.Array


# ---------------------------------------------------------------------------
# Single-trajectory factory
# ---------------------------------------------------------------------------

def make_nonequil_dds_sample_fn(
    dds_sampler,
    params,
    sigma,
    n_diffusion_steps: int,
    T_final: float,
    *,
    with_correction: bool = True,
):
    """Build a JIT-compiled single-trajectory reverse-SDE sampler.

    Parameters
    ----------
    dds_sampler       : namespace from make_nonequilibrium_dds_sampler
    params            : score-network parameters (ignored when with_correction=False)
    sigma             : diffusion coefficient σ — scalar or (D,) array
    n_diffusion_steps : number of EM steps per trajectory
    T_final           : diffusion horizon T
    with_correction   : if False, uses only the analytic reverse-OU drift (training-free)

    Returns
    -------
    sample_fn : JIT-compiled  (x0, key) -> (x_T, log_girsanov)
        x_T          : terminal sample at t = T
        log_girsanov : Σ_k [dt/2 ‖σ·f_k‖² + √dt ⟨σ·f_k, ε_k⟩]  (scalar)
                       Zero when with_correction=False.
    """
    ts    = jnp.linspace(0.0, T_final, n_diffusion_steps + 1)
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    sigma2 = sigma ** 2
    N = n_diffusion_steps

    def full_drift(x, t):
        d = dds_sampler.reverse_ou_drift(x, t)
        if with_correction:
            d = d + sigma2 * dds_sampler.score_correction_fn(x, t, params)
        return d

    @jax.jit
    def sample_fn(x0: Array, key: Array) -> tuple[Array, Array]:
        noise_keys = jr.split(key, N)

        def step(x, inp):
            t_n, t_np1, nk = inp
            dt = t_np1 - t_n
            eps = jr.normal(nk, x.shape)

            f_val = (dds_sampler.score_correction_fn(x, t_n, params)
                     if with_correction else jnp.zeros_like(x))
            drift = dds_sampler.reverse_ou_drift(x, t_n) + sigma2 * f_val
            x_new = x + dt * drift + sigma * jnp.sqrt(dt) * eps

            # Girsanov log-ratio increment
            quad  = dt / 2.0 * jnp.sum(sigma2 * f_val ** 2)
            noise = jnp.sqrt(dt) * jnp.sum(sigma * f_val * eps)
            return x_new, quad + noise

        x_T, increments = jax.lax.scan(step, x0, (ts[:-1], ts[1:], noise_keys))
        return x_T, jnp.sum(increments)

    return sample_fn


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_nonequil_dds(
    dds_sampler,
    params,
    sigma,
    n_samples: int,
    n_diffusion_steps: int,
    T_final: float,
    rng_key: Array,
    *,
    log_target: Optional[Callable] = None,
    with_correction: bool = True,
) -> tuple[Array, Array]:
    """Draw n_samples independent nonequilibrium DDS trajectories.

    Each trajectory starts from x_0 ~ N(0, σ²I) and runs the reverse-OU SDE
    to t = T.  All trajectories are independent (no MCMC).

    Parameters
    ----------
    dds_sampler       : namespace from make_nonequilibrium_dds_sampler
    params            : score-network parameters (pass None when with_correction=False)
    sigma             : diffusion coefficient σ — scalar or (D,) array
    n_samples         : number of independent output samples
    n_diffusion_steps : number of EM steps per trajectory
    T_final           : diffusion horizon T
    rng_key           : JAX PRNGKey
    log_target        : log π(x) — required for full IS weights
    with_correction   : if False, uses only the analytic reverse-OU drift

    Returns
    -------
    samples     : shape (n_samples, D)
    log_weights : shape (n_samples,)
        Full IS weights  log π(x_T) - log q_ref(x_T) - log_girsanov
        when log_target is provided, else -log_girsanov.
    """
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)

    sample_fn = make_nonequil_dds_sample_fn(
        dds_sampler, params, sigma_arr, n_diffusion_steps, T_final,
        with_correction=with_correction,
    )

    # dds_sampler.sigma is always (D,); use it to get the correct initial shape
    sigma_ref = jnp.asarray(dds_sampler.sigma, dtype=jnp.float32)
    D = sigma_ref.shape[0]

    key_init, key_traj = jr.split(rng_key)
    x0s  = sigma_ref * jr.normal(key_init, (n_samples, D))
    keys = jr.split(key_traj, n_samples)

    samples, log_girsanov = vmap(sample_fn)(x0s, keys)

    if log_target is not None:
        log_weights = (vmap(log_target)(samples)
                       - vmap(dds_sampler.log_ref_density)(samples)
                       - log_girsanov)
    else:
        log_weights = -log_girsanov

    return samples, log_weights
