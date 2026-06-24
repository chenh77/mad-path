"""
Adaptive annealed importance sampling with RWM or MALA transitions.

The base distribution is N(0, ref_std² I).  The geometric bridge at
inverse temperature β is:

    log p_β(x) = β · log π(x) + (1 − β) · log N(x; 0, ref_std² I)

Instead of a fixed annealing grid, the inverse temperature is advanced
adaptively from 0 (base) to 1 (target); see the "Adaptive AIS" section below.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

# Imported for the population-level MCMC step used by adaptive AIS.
from sampler.importance_sampling.smc import _run_bridge_population_transition  # noqa: E402

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]


def _base_log_density(x, ref_std):
    """Log density of N(0, ref_std² I)."""
    D = x.shape[0]
    return (-0.5 * jnp.sum(x ** 2) / ref_std ** 2
            - 0.5 * D * jnp.log(2 * jnp.pi * ref_std ** 2))


# ── Adaptive AIS ──────────────────────────────────────────────────────────────
# Based on the CESS bisection scheme from:
#   Zhou, Johansen & Aston (2016). Toward automatic model comparison: an
#   adaptive sequential Monte Carlo approach. JCGS 25(3): 701–726.

_N_BISECT: int = 32  # 2^{-32} ≈ 2.3e-10 precision on the temperature step


def _ais_log_cess(
    delta_beta: Array,
    log_rho: Array,
    log_weights: Array,
    num_samples: int,
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
    return jnp.log(float(num_samples)) + 2.0 * log_e1 - log_e2


def _ais_find_next_beta(
    beta: Array,
    log_rho: Array,
    log_weights: Array,
    num_samples: int,
    target_cess: float,
) -> Array:
    """
    Return beta_next in [beta, 1.0] via bisection targeting
    CESS = target_cess × num_samples.
    """
    max_delta = 1.0 - beta
    log_target = jnp.log(target_cess * float(num_samples))

    log_cess_full = _ais_log_cess(max_delta, log_rho, log_weights, num_samples)

    def bisect_step(state, _):
        lo, hi = state
        mid = (lo + hi) / 2.0
        above = _ais_log_cess(mid, log_rho, log_weights, num_samples) > log_target
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
        "num_samples",
        "num_steps",
        "max_levels",
        "transition_type",
    ),
)
def _while_adaptive_annealed_importance_sampling(
    log_density: LogDensity,
    D: int,
    ref_std: float,
    num_samples: int,
    step_size: float,
    num_steps: int,
    target_cess: float,
    max_levels: int,
    rng_key: Array,
    transition_type: str,
) -> tuple[Array, Array, Array, Array]:
    init_key, path_key = random.split(rng_key)
    level_keys = random.split(path_key, max_levels)

    states = ref_std * random.normal(init_key, shape=(num_samples, D))
    target_logps = vmap(log_density)(states)
    base_logps = vmap(lambda x: _base_log_density(x, ref_std))(states)
    log_weights = jnp.zeros(num_samples, dtype=target_logps.dtype)

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
        level_key = level_keys[level_idx]

        log_rho = target_logps - base_logps
        # Use the accumulated (possibly non-uniform) log_weights in the CESS
        # criterion so that existing weight imbalance informs the step size.
        beta_next = _ais_find_next_beta(beta, log_rho, log_weights, num_samples, target_cess)

        # Accumulate AIS weights — no resampling.
        log_weights_next = log_weights + (beta_next - beta) * log_rho

        transition_keys = random.split(level_key, num_samples)
        states_next, tlogps_next, blogps_next, accept_rate = (
            _run_bridge_population_transition(
                log_density,
                states,
                target_logps,
                base_logps,
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
            log_weights_next,
            beta_next,
            level_idx + 1,
            acc_sum + accept_rate,
        )

    final = lax.while_loop(cond_fn, body_fn, init_carry)
    states_f, _, _, lw_f, _, num_levels, acc_sum = final
    n = jnp.maximum(num_levels, 1).astype(acc_sum.dtype)
    return states_f, lw_f, acc_sum / n, num_levels


def adaptive_annealed_importance_sampling(
    log_density: LogDensity,
    D: int,
    ref_std: float,
    num_samples: int,
    step_size: float = 0.1,
    num_steps: int = 10,
    target_cess: float = 0.5,
    max_levels: int = 500,
    rng_key: Array | None = None,
    transition_type: str = "mala",
    return_info: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, dict]:
    """
    Run adaptive AIS with automatic temperature selection (Zhou et al. 2016).

    Instead of a fixed annealing grid, the next inverse temperature is found
    by bisection so that CESS = target_cess × num_samples.  Particles evolve
    independently (no resampling); the shared schedule is derived from the
    current particle cloud.  Accumulated non-uniform log-weights enter the
    CESS criterion so that existing weight imbalance informs subsequent steps.

    The base distribution is N(0, ref_std² I).

    Parameters
    ----------
    log_density    : Callable returning the target log density at a single point.
    D              : Dimension of the state space.
    ref_std        : Standard deviation of the isotropic Gaussian base.
    num_samples    : Number of AIS trajectories.
    step_size      : MCMC proposal scale at each temperature level.
    num_steps      : Number of MCMC steps per temperature level.
    target_cess    : Target CESS as a fraction of num_samples (default 0.5).
                     Larger values → more, smaller temperature steps.
    max_levels     : Hard cap on the number of temperature levels.
    rng_key        : Optional JAX PRNG key.
    transition_type: ``"rwm"`` or ``"mala"``.
    return_info    : When True return a diagnostics dict as the third element.

    Returns
    -------
    samples        : Final AIS particles approximating the target.
    log_weights    : AIS log-importance weights.
    acceptance_rate: Mean MCMC acceptance rate across all levels.
    info           : (only when return_info=True) dict with keys
                     ``acceptance_rate``, ``num_levels``, ``converged``.
    """
    if transition_type not in {"rwm", "mala"}:
        raise ValueError("transition_type must be 'rwm' or 'mala'.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
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
        _while_adaptive_annealed_importance_sampling(
            log_density,
            D,
            ref_std,
            num_samples,
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
