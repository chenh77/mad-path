"""Nonequilibrium MAD-path MCMC kernel with Euler-Maruyama discretisation.

Designed for use with the reverse-OU GMM base process trained by
dds_nonequilibrium.py.

Forward (noising) leg uses the OU forward process  dX = -½X dt + σ dW.
Backward (denoising) leg uses the reverse-OU drift (analytic for a GMM
reference) plus an optional learned correction f_θ:

    full_drift(x, t) = reverse_ou_drift(x, t) + σ² f_θ(x, t)

With with_correction=False the sampler is training-free: the score
dds_sampler.reverse_ou_drift(x, t) = ½x + σ² ∇log p_{T-t}(x) is computed
analytically from the GMM OU marginal.

Girsanov terms:
  A_k = log q_full(z | z_new, t_lo) − log q_OU(z_new | z)
  B_k = log q_OU(w | w_new)         − log q_full(w_new | w, t_lo)

Public API
----------
make_nonequil_mh_step(dds_sampler, params, sigma, log_target,
                       n_diffusion_steps, T_final, with_correction=True)
    -> step : JIT'd callable  (x0, key) -> (x_next, accepted, delta)

run_nonequil_mad_path(log_target, dds_sampler, params, sigma, T_final,
                       initial_state, num_steps, n_diffusion_steps, rng_key,
                       with_correction=True, n_mala_steps=0,
                       mala_step_size=0.01, burnin=0)
    -> (chain, acceptance_rate)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

Array = jax.Array


def make_nonequil_mh_step(
    dds_sampler,
    params,
    sigma,
    log_target: Callable,
    n_diffusion_steps: int,
    T_final: float,
    with_correction: bool = True,
):
    """Build a JIT-compiled nonequilibrium MAD-path step.

    Parameters
    ----------
    dds_sampler       : namespace returned by make_nonequilibrium_dds_sampler
    params            : score-network parameters (ignored when with_correction=False)
    sigma             : diffusion coefficient σ — scalar or (D,) array
    log_target        : log π(x)
    n_diffusion_steps : EM steps per proposal
    T_final           : diffusion horizon
    with_correction   : if False, uses only the analytic reverse-OU drift (no training needed)

    Returns
    -------
    step : JIT'd callable  (x0, key) -> (x_next, accepted, delta)
    """
    ts     = jnp.linspace(0.0, T_final, n_diffusion_steps + 1)
    sigma  = jnp.asarray(sigma, dtype=jnp.float32)   # scalar or (D,)
    sigma2 = sigma ** 2
    N      = n_diffusion_steps

    def full_drift(x, t):
        d = 0.5 * x + sigma2 * dds_sampler.ou_marginal_score(x, T_final - t)
        if with_correction:
            d = d + sigma2 * dds_sampler.score_correction_fn(x, t, params)
        return d

    @jax.jit
    def step(x0, key):
        k_eps, k_zeta, k_u = jr.split(key, 3)
        eps_all  = jr.normal(k_eps,  (N,) + x0.shape)
        zeta_all = jr.normal(k_zeta, (N,) + x0.shape)

        # Forward leg: OU forward  x0 (t=T) → z_ref (t=0)
        def noise_step(z, k):
            idx  = N - 1 - k
            t_lo = ts[idx]
            dt   = ts[idx + 1] - ts[idx]
            var  = sigma2 * dt

            z_new = z - 0.5 * dt * z + sigma * jnp.sqrt(dt) * eps_all[k]

            mean_rev    = z_new + dt * full_drift(z_new, t_lo)
            log_q_full  = -0.5 * jnp.sum((z - mean_rev) ** 2 / var)
            log_q_ou    = -0.5 * jnp.sum((z_new - z + 0.5 * dt * z) ** 2 / var)
            return z_new, log_q_full - log_q_ou

        z_ref, A_vals = jax.lax.scan(noise_step, x0, jnp.arange(N))

        # Backward leg: reverse-OU  z_ref (t=0) → y0 (t=T)
        def denoise_step(w, k):
            t_lo = ts[k]
            dt   = ts[k + 1] - ts[k]
            var  = sigma2 * dt

            drift = full_drift(w, t_lo)
            w_new = w + dt * drift + sigma * jnp.sqrt(dt) * zeta_all[k]

            mean_ou_bwd  = w_new - 0.5 * dt * w_new
            log_q_ou_bwd = -0.5 * jnp.sum((w - mean_ou_bwd) ** 2 / var)
            log_q_full   = -0.5 * jnp.sum((w_new - w - dt * drift) ** 2 / var)
            return w_new, log_q_ou_bwd - log_q_full

        y0, B_vals = jax.lax.scan(denoise_step, z_ref, jnp.arange(N))

        delta  = log_target(y0) - log_target(x0) + jnp.sum(A_vals) + jnp.sum(B_vals)
        accept = jnp.log(jr.uniform(k_u)) < jnp.minimum(delta, 0.0)
        return jnp.where(accept, y0, x0), accept, delta

    return step


def run_nonequil_mad_path(
    log_target: Callable,
    dds_sampler,
    params,
    sigma: float,
    T_final: float,
    initial_state: Array,
    num_steps: int,
    n_diffusion_steps: int,
    rng_key: Array,
    with_correction: bool = True,
    n_mala_steps: int = 0,
    mala_step_size: float = 0.01,
    burnin: int = 0,
) -> tuple[Array, float]:
    """Run a nonequilibrium MAD-path chain with optional interleaved MALA steps.

    Parameters
    ----------
    log_target        : log π(x)
    dds_sampler       : namespace from make_nonequilibrium_dds_sampler
    params            : score-network params (pass None when with_correction=False)
    sigma             : diffusion coefficient
    T_final           : diffusion horizon
    initial_state     : starting point, shape (D,)
    num_steps         : total step budget across all kernels.  Each outer
                        iteration costs (1 + n_mala_steps) steps.  Any remainder
                        steps after the last complete outer iteration are spent as
                        1 MAD-path step followed by as many MALA steps as the
                        budget allows, and that partial iteration is appended to
                        the chain.
    n_diffusion_steps : EM steps per MAD-path proposal
    rng_key           : JAX PRNGKey
    with_correction   : if False, uses only analytic reverse-OU drift (training-free)
    n_mala_steps      : number of MALA steps to run after each MAD-path step (0 = disabled)
    mala_step_size    : step size h for MALA  (proposal ~ N(x + h/2 ∇log π(x), h I))
    burnin            : number of steps to discard before collecting the chain

    Returns
    -------
    chain           : shape (n_outer [+ 1], D)  — one entry per outer iteration
    acceptance_rate : MAD-path acceptance rate (scalar float)
    """
    n_collect = num_steps - burnin
    n_outer   = n_collect // (1 + n_mala_steps)
    remainder = n_collect  % (1 + n_mala_steps)  # 0 … n_mala_steps leftover steps

    step_fn = make_nonequil_mh_step(
        dds_sampler, params, sigma, log_target,
        n_diffusion_steps, T_final, with_correction,
    )

    grad_log_target = jax.grad(log_target)

    @jax.jit
    def mala_step(x, key):
        k_prop, k_u = jr.split(key)
        g      = grad_log_target(x)
        x_prop = x + 0.5 * mala_step_size * g + jnp.sqrt(mala_step_size) * jr.normal(k_prop, x.shape)
        g_prop = grad_log_target(x_prop)

        log_q_fwd = -0.5 / mala_step_size * jnp.sum((x_prop - x - 0.5 * mala_step_size * g) ** 2)
        log_q_bwd = -0.5 / mala_step_size * jnp.sum((x - x_prop - 0.5 * mala_step_size * g_prop) ** 2)
        delta  = log_target(x_prop) - log_target(x) + log_q_bwd - log_q_fwd
        accept = jnp.log(jr.uniform(k_u)) < jnp.minimum(delta, 0.0)
        return jnp.where(accept, x_prop, x), accept

    def body(x, key):
        keys   = jr.split(key, n_mala_steps + 1)
        k_mad  = keys[0]
        k_mala = keys[1:]  # shape (n_mala_steps, 2)

        x_mad, accepted_mad, _ = step_fn(x, k_mad)

        def mala_body(x, k):
            x_next, _ = mala_step(x, k)
            return x_next, x_next  # record every MALA state

        x_final, mala_states = jax.lax.scan(mala_body, x_mad, k_mala)
        # all_states: (1 + n_mala_steps, D) — MAD-path sample + each MALA sample
        all_states = jnp.concatenate([x_mad[None], mala_states], axis=0)
        return x_final, (all_states, accepted_mad)

    # --- burn-in: run and discard steps to warm up the chain ---
    key_burnin, key_main = jr.split(rng_key)
    if burnin > 0:
        n_burnin_outer = max(1, burnin // (1 + n_mala_steps))
        burnin_keys = jr.split(key_burnin, n_burnin_outer)
        initial_state, _ = jax.lax.scan(body, initial_state, burnin_keys)

    # --- complete outer iterations ---
    n_keys   = n_outer + (1 if remainder > 0 else 0)
    all_keys = jr.split(key_main, max(n_keys, 1))

    if n_outer > 0:
        x_last, (chain_main, accepted_main) = jax.lax.scan(body, initial_state, all_keys[:n_outer])
        # chain_main: (n_outer, 1+n_mala_steps, D) → flatten to (n_outer*(1+n_mala_steps), D)
        chain_main = chain_main.reshape(-1, *initial_state.shape)
    else:
        x_last        = initial_state
        chain_main    = jnp.empty((0,) + initial_state.shape)
        accepted_main = jnp.zeros((0,), dtype=jnp.bool_)

    # --- partial last iteration (remainder steps) ---
    if remainder > 0:
        rem_keys  = jr.split(all_keys[n_outer], remainder)
        x_last, accepted_mad, _ = step_fn(x_last, rem_keys[0])
        rem_states = [x_last]
        for i in range(1, remainder):          # up to remainder-1 MALA steps
            x_last, _ = mala_step(x_last, rem_keys[i])
            rem_states.append(x_last)
        chain        = jnp.concatenate([chain_main, jnp.stack(rem_states)], axis=0)
        accepted_all = jnp.concatenate([accepted_main, jnp.array([accepted_mad])], axis=0)
    else:
        chain        = chain_main
        accepted_all = accepted_main

    return chain, float(jnp.mean(accepted_all))
