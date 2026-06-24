"""
Adaptive annealed sequential Monte Carlo with RWM or MALA transitions.

The base distribution is N(0, ref_std² I).  The geometric bridge at
inverse temperature β is:

    log p_β(x) = β · log π(x) + (1 − β) · log N(x; 0, ref_std² I)

Instead of a fixed annealing grid, the inverse temperature is advanced
adaptively from 0 (base) to 1 (target); see the "Adaptive SMC" section below.
Particles are resampled (systematically) after every intermediate weight
update; final target-level weights are retained in the returned
``log_weights``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

from sampler.mcmc.mala import run_mala_steps
from sampler.mcmc.rwm import run_rwm_steps

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]


def _base_log_density(x, ref_std):
    """Log density of N(0, ref_std² I)."""
    D = x.shape[0]
    return (-0.5 * jnp.sum(x ** 2) / ref_std ** 2
            - 0.5 * D * jnp.log(2 * jnp.pi * ref_std ** 2))


def _bridge_log_density(beta: Array, target_logp: Array, base_logp: Array) -> Array:
    return beta * target_logp + (1.0 - beta) * base_logp


def _systematic_resample_indices(key: Array, log_weights: Array) -> Array:
    """Return systematic-resampling ancestor indices."""
    num_particles = log_weights.shape[0]
    shifted = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(shifted)
    cumulative = jnp.cumsum(weights)
    u0 = random.uniform(key, (), dtype=weights.dtype) / num_particles
    positions = u0 + jnp.arange(num_particles, dtype=weights.dtype) / num_particles
    indices = jnp.searchsorted(cumulative, positions)
    return jnp.clip(indices, 0, num_particles - 1)


@partial(jit, static_argnames=("log_density", "num_steps", "transition_type"))
def _run_bridge_population_transition(
    log_density: LogDensity,
    states: Array,
    target_logps: Array,
    base_logps: Array,
    beta: Array,
    step_size: float,
    ref_std: float,
    num_steps: int,
    rng_keys: Array,
    transition_type: str,
) -> tuple[Array, Array, Array, Array]:
    """Apply the chosen Markov kernel to every particle at bridge level β."""

    def bridge_log_density(state):
        target_logp_state = log_density(state)
        base_logp_state = _base_log_density(state, ref_std)
        return _bridge_log_density(beta, target_logp_state, base_logp_state)

    initial_bridge_logps = _bridge_log_density(beta, target_logps, base_logps)

    if transition_type == "rwm":
        next_states, _, accept_rates = vmap(
            lambda x, lp, key: run_rwm_steps(
                bridge_log_density,
                x,
                step_size=step_size,
                num_steps=num_steps,
                rng_key=key,
                initial_logp=lp,
            )
        )(states, initial_bridge_logps, rng_keys)
    else:
        next_states, _, accept_rates = vmap(
            lambda x, lp, key: run_mala_steps(
                bridge_log_density,
                x,
                step_size=step_size,
                num_steps=num_steps,
                rng_key=key,
                initial_logp=lp,
            )
        )(states, initial_bridge_logps, rng_keys)

    next_target_logps = vmap(log_density)(next_states)
    next_base_logps = vmap(lambda x: _base_log_density(x, ref_std))(next_states)
    return next_states, next_target_logps, next_base_logps, jnp.mean(accept_rates)


# ── Adaptive SMC ──────────────────────────────────────────────────────────────
# Based on the CESS bisection scheme from:
#   Zhou, Johansen & Aston (2016). Toward automatic model comparison: an
#   adaptive sequential Monte Carlo approach. JCGS 25(3): 701–726.

_N_BISECT: int = 32  # 2^{-32} ≈ 2.3e-10 precision on the temperature step


def _smc_log_cess(
    delta_beta: Array,
    log_rho: Array,
    log_weights: Array,
    num_particles: int,
) -> Array:
    """
    log CESS_N(delta_beta) for a geometric bridge increment.

    CESS_N = N · [E_w[α]]²  /  E_w[α²],   α_i = exp(delta_beta · log_rho_i)

    Uses unnormalized log_weights for numerical stability.
    """
    lw = log_weights - jax.scipy.special.logsumexp(log_weights)
    log_incr = delta_beta * log_rho
    log_e1 = jax.scipy.special.logsumexp(lw + log_incr)
    log_e2 = jax.scipy.special.logsumexp(lw + 2.0 * log_incr)
    return jnp.log(float(num_particles)) + 2.0 * log_e1 - log_e2


def _smc_find_next_beta(
    beta: Array,
    log_rho: Array,
    log_weights: Array,
    num_particles: int,
    target_cess: float,
) -> Array:
    """
    Return beta_next in [beta, 1.0] via bisection targeting
    CESS = target_cess × num_particles.
    """
    max_delta = 1.0 - beta
    log_target = jnp.log(target_cess * float(num_particles))

    log_cess_full = _smc_log_cess(max_delta, log_rho, log_weights, num_particles)

    def bisect_step(state, _):
        lo, hi = state
        mid = (lo + hi) / 2.0
        above = _smc_log_cess(mid, log_rho, log_weights, num_particles) > log_target
        return (jnp.where(above, mid, lo), jnp.where(above, hi, mid)), None

    (lo, hi), _ = lax.scan(
        bisect_step,
        (jnp.zeros_like(max_delta), max_delta),
        None,
        length=_N_BISECT,
    )
    delta_bisect = (lo + hi) / 2.0
    delta_beta = jnp.where(log_cess_full >= log_target, max_delta, delta_bisect)
    return jnp.minimum(beta + delta_beta, jnp.ones_like(beta))


@partial(
    jit,
    static_argnames=(
        "log_density",
        "D",
        "num_particles",
        "num_steps",
        "max_levels",
        "transition_type",
    ),
)
def _while_adaptive_sequential_monte_carlo(
    log_density: LogDensity,
    D: int,
    ref_std: float,
    num_particles: int,
    step_size: float,
    num_steps: int,
    target_cess: float,
    max_levels: int,
    rng_key: Array,
    transition_type: str,
) -> tuple[Array, Array, Array, Array]:
    init_key, path_key = random.split(rng_key)
    level_keys = random.split(path_key, max_levels)

    states = ref_std * random.normal(init_key, shape=(num_particles, D))
    target_logps = vmap(log_density)(states)
    base_logps = vmap(lambda x: _base_log_density(x, ref_std))(states)
    log_weights = jnp.zeros(num_particles, dtype=target_logps.dtype)

    init_carry = (
        states,
        target_logps,
        base_logps,
        log_weights,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0.0, dtype=target_logps.dtype),
    )

    def cond_fn(carry):
        _, _, _, _, beta, level_idx, _ = carry
        return jnp.logical_and(beta < 1.0 - 1e-6, level_idx < max_levels)

    def body_fn(carry):
        states, target_logps, base_logps, log_weights, beta, level_idx, acc_sum = carry
        resample_key, transition_key = random.split(level_keys[level_idx])

        log_rho = target_logps - base_logps
        beta_next = _smc_find_next_beta(beta, log_rho, log_weights, num_particles, target_cess)

        log_weights_updated = log_weights + (beta_next - beta) * log_rho
        at_target = beta_next >= 1.0 - 1e-6

        def resample_pop(_):
            idx = _systematic_resample_indices(resample_key, log_weights_updated)
            return (
                states[idx],
                target_logps[idx],
                base_logps[idx],
                jnp.zeros_like(log_weights_updated),
            )

        def keep_pop(_):
            return states, target_logps, base_logps, log_weights_updated

        # Resample at every intermediate level; retain weights at the target.
        states_in, tlogps_in, blogps_in, lw_in = lax.cond(
            ~at_target, resample_pop, keep_pop, operand=None
        )

        transition_keys = random.split(transition_key, num_particles)
        states_next, tlogps_next, blogps_next, accept_rate = (
            _run_bridge_population_transition(
                log_density,
                states_in,
                tlogps_in,
                blogps_in,
                beta_next,
                step_size,
                ref_std,
                num_steps,
                transition_keys,
                transition_type,
            )
        )

        return (
            states_next,
            tlogps_next,
            blogps_next,
            lw_in,
            beta_next,
            level_idx + 1,
            acc_sum + accept_rate,
        )

    final = lax.while_loop(cond_fn, body_fn, init_carry)
    states_f, _, _, lw_f, _, num_levels, acc_sum = final
    n = jnp.maximum(num_levels, 1).astype(acc_sum.dtype)
    return states_f, lw_f, acc_sum / n, num_levels


def adaptive_sequential_monte_carlo(
    log_density: LogDensity,
    D: int,
    ref_std: float,
    num_particles: int,
    step_size: float = 0.1,
    num_steps: int = 10,
    target_cess: float = 0.5,
    max_levels: int = 500,
    rng_key: Array | None = None,
    transition_type: str = "mala",
    return_info: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, dict]:
    """
    Run adaptive annealed SMC with automatic temperature selection (Zhou et al. 2016).

    Instead of a fixed annealing grid, the next inverse temperature is found
    by bisection so that CESS = target_cess × num_particles.  Particles are
    resampled (systematically) after every intermediate weight update, exactly
    as in Zhou et al. (2016).  Weights at the final target level are retained.

    The base distribution is N(0, ref_std² I).

    Parameters
    ----------
    log_density    : Callable returning the target log density at a single point.
    D              : Dimension of the state space.
    ref_std        : Standard deviation of the isotropic Gaussian base.
    num_particles  : Number of SMC particles.
    step_size      : MCMC proposal scale at each temperature level.
    num_steps      : Number of MCMC steps per temperature level.
    target_cess    : Target CESS as a fraction of num_particles (default 0.5).
                     Larger values → more, smaller temperature steps.
    max_levels     : Hard cap on the number of temperature levels.
    rng_key        : Optional JAX PRNG key.
    transition_type: ``"rwm"`` or ``"mala"``.
    return_info    : When True return a diagnostics dict as the third element.

    Returns
    -------
    samples        : Final particles approximating the target.
    log_weights    : Final log-weights (meaningful; no resampling at target).
    acceptance_rate: Mean MCMC acceptance rate across all levels.
    info           : (only when return_info=True) dict with keys
                     ``acceptance_rate``, ``num_levels``, ``converged``.
    """
    if transition_type not in {"rwm", "mala"}:
        raise ValueError("transition_type must be 'rwm' or 'mala'.")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0.0 < target_cess < 1.0:
        raise ValueError("target_cess must lie in (0, 1).")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive.")

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    samples, log_weights, acceptance_rate, num_levels = (
        _while_adaptive_sequential_monte_carlo(
            log_density,
            D,
            ref_std,
            num_particles,
            step_size,
            num_steps,
            target_cess,
            max_levels,
            rng_key,
            transition_type,
        )
    )

    if not return_info:
        return samples, log_weights, acceptance_rate

    info = {
        "acceptance_rate": acceptance_rate,
        "num_levels": num_levels,
        "converged": num_levels < max_levels,
    }
    return samples, log_weights, info
