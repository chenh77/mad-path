"""
Neal's tempered transitions
https://link.springer.com/article/10.1007/BF00143556
"""

from collections.abc import Callable, Sequence
from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from sampler.mcmc.mala import run_mala_steps
from sampler.mcmc.rwm import run_rwm_steps

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]


@partial(
    jit,
    static_argnames=("log_density", "num_iterations", "num_steps", "transition_type"),
)
def _scan_tempered_transition(
    log_density: LogDensity,
    initial_state: Array,
    initial_base_logp: Array,
    inverse_temperatures: Array,
    step_sizes: Array,
    num_iterations: int,
    num_steps: int,
    rng_key: Array,
    transition_type: str,
) -> tuple[Array, Array]:
    num_levels = inverse_temperatures.shape[0] - 1
    keys = random.split(rng_key, num_iterations)

    def one_iteration(carry, key):
        x, base_logp_x = carry
        path_key, accept_key = random.split(key)
        level_keys = random.split(path_key, 2 * num_levels)

        upward_states = [x]
        upward_base_logps = [base_logp_x]
        current = x
        current_base_logp = base_logp_x

        for level in range(1, inverse_temperatures.shape[0]):
            beta = inverse_temperatures[level]
            if transition_type == "rwm":
                current, scaled_logp, _ = run_rwm_steps(
                    log_density,
                    current,
                    step_size=step_sizes[level],
                    num_steps=num_steps,
                    rng_key=level_keys[level - 1],
                    initial_logp=beta * current_base_logp,
                    log_density_scale=beta,
                )
            else:
                current, scaled_logp, _ = run_mala_steps(
                    log_density,
                    current,
                    step_size=step_sizes[level],
                    num_steps=num_steps,
                    rng_key=level_keys[level - 1],
                    initial_logp=beta * current_base_logp,
                    log_density_scale=beta,
                )
            current_base_logp = scaled_logp / beta
            upward_states.append(current)
            upward_base_logps.append(current_base_logp)

        downward_states = [upward_states[-1]]
        downward_base_logps = [upward_base_logps[-1]]
        current = upward_states[-1]
        current_base_logp = upward_base_logps[-1]

        for idx in range(num_levels):
            level = num_levels - idx
            beta = inverse_temperatures[level]
            if transition_type == "rwm":
                current, scaled_logp, _ = run_rwm_steps(
                    log_density,
                    current,
                    step_size=step_sizes[level],
                    num_steps=num_steps,
                    rng_key=level_keys[num_levels + idx],
                    initial_logp=beta * current_base_logp,
                    log_density_scale=beta,
                )
            else:
                current, scaled_logp, _ = run_mala_steps(
                    log_density,
                    current,
                    step_size=step_sizes[level],
                    num_steps=num_steps,
                    rng_key=level_keys[num_levels + idx],
                    initial_logp=beta * current_base_logp,
                    log_density_scale=beta,
                )
            current_base_logp = scaled_logp / beta
            downward_states.append(current)
            downward_base_logps.append(current_base_logp)

        proposal = downward_states[-1]
        proposal_base_logp = downward_base_logps[-1]
        downward_base_logps = list(reversed(downward_base_logps[1:]))

        log_alpha = 0.0
        for level in range(num_levels):
            beta = inverse_temperatures[level]
            beta_next = inverse_temperatures[level + 1]
            up_base_logp = upward_base_logps[level]
            down_base_logp = downward_base_logps[level]
            log_alpha += (beta_next - beta) * up_base_logp
            log_alpha += (beta - beta_next) * down_base_logp

        accepted = jnp.log(random.uniform(accept_key)) < log_alpha
        x_next = jnp.where(accepted, proposal, x)
        base_logp_next = jnp.where(accepted, proposal_base_logp, base_logp_x)
        return (x_next, base_logp_next), (x_next, accepted)

    (_, _), (states, accepted) = lax.scan(
        one_iteration, (initial_state, initial_base_logp), keys
    )
    return states, accepted


def tempered_transition(
    log_density: LogDensity,
    initial_state: Array,
    inverse_temperature_schedule: Sequence[float],
    num_iterations: int,
    burnin: int = 0,
    thinning: int = 1,
    step_size: float | Sequence[float] = 0.1,
    num_steps: int = 10,
    rng_key: Array | None = None,
    transition_type: str = "rwm",
) -> tuple[Array, Array]:
    """
    Run Neal's tempered transition MCMC using either RWM or MALA at every
    inverse-temperature level.

    ``step_size`` may be a scalar (applied uniformly across all levels) or a
    sequence of length equal to ``inverse_temperature_schedule`` (one value per
    level, matched by index).  The latter lets you pass per-level step sizes
    obtained from adaptive parallel tempering::

        _, apt_diag = adaptive_parallel_tempering(...)
        tempered_transition(
            log_density, x0,
            inverse_temperature_schedule=apt_diag["final_inverse_temperatures"],
            step_size=apt_diag["final_step_sizes"],
            ...
        )
    """

    inverse_temperatures = [float(beta) for beta in inverse_temperature_schedule]
    n_levels = len(inverse_temperatures)
    if transition_type not in {"rwm", "mala"}:
        raise ValueError("transition_type must be either 'rwm' or 'mala'.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if burnin < 0 or burnin >= num_iterations:
        raise ValueError("burnin must satisfy 0 <= burnin < num_iterations.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    x0 = jnp.asarray(initial_state)

    try:
        step_size_scalar = float(step_size)  # type: ignore[arg-type]
        step_sizes_list = [step_size_scalar] * n_levels
    except (TypeError, ValueError):
        step_sizes_list = [float(s) for s in step_size]  # type: ignore[union-attr]
        if len(step_sizes_list) != n_levels:
            raise ValueError(
                f"step_size sequence length ({len(step_sizes_list)}) must match "
                f"inverse_temperature_schedule length ({n_levels})."
            )
    if any(s <= 0.0 for s in step_sizes_list):
        raise ValueError("All step_size values must be positive.")

    base_logp0 = log_density(x0)
    inverse_temperatures_array = jnp.asarray(inverse_temperatures, dtype=x0.dtype)
    step_sizes_array = jnp.asarray(step_sizes_list, dtype=x0.dtype)
    states, accepted = _scan_tempered_transition(
        log_density,
        x0,
        base_logp0,
        inverse_temperatures_array,
        step_sizes_array,
        num_iterations,
        num_steps,
        rng_key,
        transition_type,
    )
    samples = states[burnin::thinning]
    acceptance_rate = jnp.mean(accepted)
    return samples, acceptance_rate
