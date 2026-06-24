from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, lax, random

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]


def _scaled_log_density(
    log_density: LogDensity, x: Array, log_density_scale: float
) -> Array:
    return log_density_scale * log_density(x)


@partial(jit, static_argnames=("log_density",))
def mala_step(
    key: Array,
    x: Array,
    logp_x: Array,
    grad_logp_x: Array,
    log_density: LogDensity,
    step_size: float,
    log_density_scale: float = 1.0,
) -> tuple[Array, Array, Array, Array]:
    proposal_key, accept_key = random.split(key)
    noise = random.normal(proposal_key, shape=x.shape)
    drift_x = 0.5 * step_size**2 * grad_logp_x
    proposal = x + drift_x + step_size * noise

    logp_proposal = _scaled_log_density(log_density, proposal, log_density_scale)
    grad_logp_proposal = log_density_scale * grad(log_density)(proposal)
    drift_proposal = 0.5 * step_size**2 * grad_logp_proposal

    log_q_forward = -0.5 * jnp.sum((proposal - x - drift_x) ** 2) / step_size**2
    log_q_backward = (
        -0.5 * jnp.sum((x - proposal - drift_proposal) ** 2) / step_size**2
    )
    log_alpha = logp_proposal + log_q_backward - logp_x - log_q_forward

    accepted = jnp.log(random.uniform(accept_key)) < log_alpha
    x_next = jnp.where(accepted, proposal, x)
    logp_next = jnp.where(accepted, logp_proposal, logp_x)
    grad_logp_next = jnp.where(accepted, grad_logp_proposal, grad_logp_x)
    return x_next, logp_next, grad_logp_next, accepted


@partial(jit, static_argnames=("log_density", "num_steps"))
def _scan_mala(
    key: Array,
    initial_state: Array,
    initial_logp: Array,
    initial_grad_logp: Array,
    log_density: LogDensity,
    step_size: float,
    num_steps: int,
    log_density_scale: float,
) -> tuple[Array, Array, Array]:
    keys = random.split(key, num_steps)

    def body(carry, step_key):
        x, logp, grad_logp = carry
        x_next, logp_next, grad_logp_next, accepted = mala_step(
            step_key,
            x,
            logp,
            grad_logp,
            log_density,
            step_size,
            log_density_scale,
        )
        return (x_next, logp_next, grad_logp_next), (x_next, logp_next, accepted)

    (_, _, _), (states, logps, accepted) = lax.scan(
        body, (initial_state, initial_logp, initial_grad_logp), keys
    )
    return states, logps, accepted


def run_mala_steps(
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
        _scaled_log_density(log_density, x0, log_density_scale)
        if initial_logp is None
        else initial_logp
    )
    grad_logp0 = log_density_scale * grad(log_density)(x0)
    states, logps, accepted = _scan_mala(
        rng_key,
        x0,
        logp0,
        grad_logp0,
        log_density,
        step_size,
        num_steps,
        log_density_scale,
    )
    return states[-1], logps[-1], jnp.mean(accepted)
