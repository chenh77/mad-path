from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]


@partial(jit, static_argnames=("log_density",))
def rwm_step(
    key: Array,
    x: Array,
    logp_x: Array,
    log_density: LogDensity,
    step_size: float,
    log_density_scale: float = 1.0,
) -> tuple[Array, Array, Array]:
    proposal_key, accept_key = random.split(key)
    proposal = x + step_size * random.normal(proposal_key, shape=x.shape)
    logp_proposal = log_density_scale * log_density(proposal)
    log_alpha = logp_proposal - logp_x
    accepted = jnp.log(random.uniform(accept_key)) < log_alpha
    x_next = jnp.where(accepted, proposal, x)
    logp_next = jnp.where(accepted, logp_proposal, logp_x)
    return x_next, logp_next, accepted


@partial(jit, static_argnames=("log_density", "num_steps"))
def _scan_rwm(
    key: Array,
    initial_state: Array,
    initial_logp: Array,
    log_density: LogDensity,
    step_size: float,
    num_steps: int,
    log_density_scale: float,
) -> tuple[Array, Array, Array]:
    keys = random.split(key, num_steps)

    def body(carry, step_key):
        x, logp = carry
        x_next, logp_next, accepted = rwm_step(
            step_key, x, logp, log_density, step_size, log_density_scale
        )
        return (x_next, logp_next), (x_next, logp_next, accepted)

    (_, _), (states, logps, accepted) = lax.scan(
        body, (initial_state, initial_logp), keys
    )
    return states, logps, accepted


def run_rwm_steps(
    log_density: LogDensity,
    initial_state: Array,
    step_size: float,
    num_steps: int,
    rng_key: Array | None = None,
    initial_logp: Array | None = None,
    log_density_scale: float = 1.0,
) -> tuple[Array, Array, Array]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    x0 = jnp.asarray(initial_state)
    logp0 = (
        log_density_scale * log_density(x0) if initial_logp is None else initial_logp
    )
    states, logps, accepted = _scan_rwm(
        rng_key,
        x0,
        logp0,
        log_density,
        step_size,
        num_steps,
        log_density_scale,
    )
    return states[-1], logps[-1], jnp.mean(accepted)
