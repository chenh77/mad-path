"""Non-equilibrium DDS with a reverse-OU GMM base process.

Replaces the equilibrium Gaussian reference of standard DDS with the reverse
process of an OU diffusion started from a user-specified Gaussian mixture
model. The resulting base process approximately maps ``N(0, sigma^2 I)`` to
that GMM over the time horizon ``[0, T_FINAL]``.

Provides ``make_nonequilibrium_dds_sampler(...)`` which returns a namespace of
JAX-friendly functions for training a score correction and sampling from the
controlled SDE.
"""

import types
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap

from sampler.vi.utils import (
    tree_weighted_mean, tree_mean, tree_add,
)


LOG_2PI = jnp.log(2.0 * jnp.pi)


# ============================================================
# Config
# ============================================================
@dataclass
class NonequilibriumDDSConfig:
    T_FINAL: float = 5.0
    D: int = 2
    n_particles: int = 256
    hidden: int = 64
    score_parametrization: str = "target_augmented"  # "target_augmented" | "base"
    detach_target_score: bool = False
    sigma: float = 1.0
    cov_regularization: float = 1e-6


# ============================================================
# MLP utilities
# ============================================================
def init_mlp(key, widths, last_scale=1.0):
    keys = jr.split(key, len(widths) - 1)
    layers = []
    for idx, (din, dout) in enumerate(zip(widths[:-1], widths[1:])):
        scale = 1.0 / jnp.sqrt(float(din))
        if idx == len(widths) - 2:
            scale = scale * last_scale
        w = scale * jr.normal(keys[idx], (din, dout), dtype=jnp.float32)
        b = jnp.zeros((dout,), dtype=jnp.float32)
        layers.append({"w": w, "b": b})
    return tuple(layers)


def mlp_apply(layers, x):
    h = x
    for idx, layer in enumerate(layers):
        h = h @ layer["w"] + layer["b"]
        if idx < len(layers) - 1:
            h = jax.nn.silu(h)
    return h


def _canonicalize_covariances(gmm_covs, num_components, dim, eye):
    covs = jnp.asarray(gmm_covs, dtype=jnp.float32)
    if covs.ndim == 1:
        if covs.shape != (num_components,):
            raise ValueError(
                "Scalar GMM covariances must have shape (K,). "
                f"Got {covs.shape} for K={num_components}."
            )
        return covs[:, None, None] * eye[None, :, :]
    if covs.ndim == 2:
        if covs.shape == (num_components, dim):
            return vmap(jnp.diag)(covs)
        raise ValueError(
            "Diagonal GMM covariances must have shape (K, D). "
            f"Got {covs.shape} for K={num_components}, D={dim}."
        )
    if covs.ndim == 3:
        if covs.shape != (num_components, dim, dim):
            raise ValueError(
                "Full GMM covariances must have shape (K, D, D). "
                f"Got {covs.shape} for K={num_components}, D={dim}."
            )
        return covs
    raise ValueError(
        "gmm_covs must have shape (K,), (K, D), or (K, D, D). "
        f"Got {covs.shape}."
    )


def _covariance_factors(covs, eps, eye):
    covs_reg = covs + eps * eye[None, :, :]
    chols = jnp.linalg.cholesky(covs_reg)
    log_dets = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(chols, axis1=-2, axis2=-1)), axis=-1
    )
    return chols, log_dets


def _gaussian_component_stats_batch(x, means, chols, log_dets, dim):
    diff = x[:, None, :] - means[None, :, :]
    diff_by_component = jnp.swapaxes(diff, 0, 1)

    def solve_one(chol, rhs):
        whitened = jax.scipy.linalg.solve_triangular(chol, rhs.T, lower=True).T
        precision_rhs = jax.scipy.linalg.solve_triangular(
            chol.T, whitened.T, lower=False
        ).T
        return whitened, precision_rhs

    whitened, precision_diff = vmap(solve_one)(chols, diff_by_component)
    mahal = jnp.sum(whitened ** 2, axis=-1).T
    precision_diff = jnp.transpose(precision_diff, (1, 0, 2))
    comp_logpdf = -0.5 * (dim * LOG_2PI + log_dets[None, :] + mahal)
    return comp_logpdf, precision_diff


def _gmm_log_density_and_score_batch(x, means, chols, log_dets, log_weights, dim):
    comp_logpdf, precision_diff = _gaussian_component_stats_batch(
        x, means, chols, log_dets, dim
    )
    log_joint = log_weights[None, :] + comp_logpdf
    log_density = jax.scipy.special.logsumexp(log_joint, axis=1)
    responsibilities = jnp.exp(log_joint - log_density[:, None])
    score = -jnp.sum(responsibilities[:, :, None] * precision_diff, axis=1)
    return log_density, score


# ============================================================
# Sampler factory
# ============================================================
def make_nonequilibrium_dds_sampler(
    log_target_density,
    gmm_means,
    gmm_covs,
    gmm_weights,
    config: NonequilibriumDDSConfig,
):
    """Build DDS functions with a reverse-OU GMM base process.

    Parameters
    ----------
    log_target_density : callable ``(x,) -> scalar``
        Target log-density.
    gmm_means : array, shape ``(K, D)``
        Means of the reference GMM.
    gmm_covs : array
        Reference GMM covariances. Accepted shapes are:
        ``(K,)`` for isotropic variances,
        ``(K, D)`` for diagonal covariances,
        ``(K, D, D)`` for full covariances.
    gmm_weights : array, shape ``(K,)``
        Mixture weights. They will be normalized internally.
    config : NonequilibriumDDSConfig

    Returns
    -------
    ``types.SimpleNamespace`` with:

    - ``init_params(key) -> params``
    - ``make_pathwise_fns(n_steps)``
    - ``make_stopgrad_fns(n_steps)``
    - ``make_reinforce_fns(n_steps)``
    - ``generate_samples(params, key, n_particles, n_steps)``
    - ``generate_base_samples(key, n_particles, n_steps)``
    """
    means = jnp.asarray(gmm_means, dtype=jnp.float32)
    if means.ndim != 2:
        raise ValueError(f"gmm_means must have shape (K, D). Got {means.shape}.")

    num_components, dim = means.shape
    if config.D != dim:
        raise ValueError(
            f"Config dimension D={config.D} does not match gmm_means shape {means.shape}."
        )

    eye = jnp.eye(dim, dtype=jnp.float32)
    covs = _canonicalize_covariances(gmm_covs, num_components, dim, eye)

    weights = jnp.asarray(gmm_weights, dtype=jnp.float32)
    if weights.shape != (num_components,):
        raise ValueError(
            f"gmm_weights must have shape ({num_components},). Got {weights.shape}."
        )
    log_weights = jnp.log(jnp.clip(weights, 1e-12))
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

    gmm_chols, gmm_log_dets = _covariance_factors(
        covs, config.cov_regularization, eye
    )

    D = config.D
    N_PARTICLES = config.n_particles
    T_FINAL = config.T_FINAL
    HIDDEN = config.hidden
    # Broadcast scalar or (D,) array → always (D,) so per-dim σ is supported.
    SIGMA = jnp.broadcast_to(
        jnp.asarray(config.sigma, dtype=jnp.float32) * jnp.ones(D, dtype=jnp.float32),
        (D,),
    )
    SIGMA2 = SIGMA ** 2
    SCORE_PARAM = config.score_parametrization
    DETACH_TGT = config.detach_target_score

    # ----------------------------------------------------------
    # GMM / OU helpers
    # ----------------------------------------------------------
    def log_ref_density_batch(x):
        log_density, _ = _gmm_log_density_and_score_batch(
            x, means, gmm_chols, gmm_log_dets, log_weights, D
        )
        return log_density

    def log_ref_density(x):
        return log_ref_density_batch(x[None, :])[0]

    def ou_marginal_params(s):
        decay = jnp.exp(-0.5 * s)
        decay_sq = jnp.exp(-s)
        means_s = decay * means
        covs_s = decay_sq * covs + (1.0 - decay_sq) * jnp.diag(SIGMA2)[None, :, :]
        return means_s, covs_s

    def ou_marginal_factors(s):
        means_s, covs_s = ou_marginal_params(s)
        chols_s, log_dets_s = _covariance_factors(
            covs_s, config.cov_regularization, eye
        )
        return means_s, chols_s, log_dets_s

    def log_ou_marginal_batch(x, s):
        means_s, chols_s, log_dets_s = ou_marginal_factors(s)
        log_density, _ = _gmm_log_density_and_score_batch(
            x, means_s, chols_s, log_dets_s, log_weights, D
        )
        return log_density

    def log_ou_marginal(x, s):
        return log_ou_marginal_batch(x[None, :], s)[0]

    def ou_marginal_score_batch(x, s):
        means_s, chols_s, log_dets_s = ou_marginal_factors(s)
        _, score = _gmm_log_density_and_score_batch(
            x, means_s, chols_s, log_dets_s, log_weights, D
        )
        return score

    def ou_marginal_score(x, s):
        return ou_marginal_score_batch(x[None, :], s)[0]

    def reverse_ou_drift_batch(x, t):
        s = T_FINAL - t
        return 0.5 * x + SIGMA2 * ou_marginal_score_batch(x, s)

    def reverse_ou_drift(x, t):
        return reverse_ou_drift_batch(x[None, :], t)[0]

    def prepare_discretization(n_steps):
        ts = jnp.linspace(0.0, T_FINAL, n_steps + 1)
        ss = T_FINAL - ts[:-1]
        means_by_step, chols_by_step, log_dets_by_step = vmap(ou_marginal_factors)(ss)
        return ts, means_by_step, chols_by_step, log_dets_by_step

    def sample_initial(key, n_particles):
        return SIGMA * jr.normal(key, (n_particles, D))

    # ----------------------------------------------------------
    # Score network
    # ----------------------------------------------------------
    def target_score_fn(x):
        return grad(log_target_density)(x)

    def time_feature(t):
        return jnp.array([2.0 * t / T_FINAL - 1.0], dtype=jnp.float32)

    def init_params(key):
        if SCORE_PARAM == "target_augmented":
            k1, k2 = jr.split(key)
            return {
                "state_mlp": init_mlp(k1, [D + 1, HIDDEN, HIDDEN, D], last_scale=1e-2),
                "gate_mlp": init_mlp(k2, [1, HIDDEN, HIDDEN, D], last_scale=1e-2),
            }
        return {
            "state_mlp": init_mlp(key, [D + 1, HIDDEN, HIDDEN, D], last_scale=1e-2),
        }

    def score_correction_fn(x, t, params):
        t_feat = time_feature(t)
        base = mlp_apply(params["state_mlp"], jnp.concatenate([x, t_feat], axis=0))
        if SCORE_PARAM == "base":
            return base
        gate = mlp_apply(params["gate_mlp"], t_feat)
        tgt_score = target_score_fn(x)
        if DETACH_TGT:
            tgt_score = jax.lax.stop_gradient(tgt_score)
        return base + gate * tgt_score

    # ----------------------------------------------------------
    # Transition / costs
    # ----------------------------------------------------------
    def _transition_batch_with_factors(
        x, eps, t, t_next, params, means_s, chols_s, log_dets_s
    ):
        dt = t_next - t
        _, base_scores = _gmm_log_density_and_score_batch(
            x, means_s, chols_s, log_dets_s, log_weights, D
        )
        base_drift = 0.5 * x + SIGMA2 * base_scores
        f_vals = vmap(score_correction_fn, (0, None, None))(x, t, params)
        drift = base_drift + SIGMA2 * f_vals
        x_new = x + drift * dt + SIGMA * jnp.sqrt(dt) * eps
        return x_new, f_vals

    def transition_batch(x, eps, t, t_next, params):
        means_s, chols_s, log_dets_s = ou_marginal_factors(T_FINAL - t)
        return _transition_batch_with_factors(
            x, eps, t, t_next, params, means_s, chols_s, log_dets_s
        )

    def step_cost_one(x, t, t_next, params):
        dt = t_next - t
        f_val = score_correction_fn(x, t, params)
        return dt / 2.0 * jnp.sum(SIGMA2 * f_val ** 2)

    def terminal_cost_batch(x):
        return log_ref_density_batch(x) - vmap(log_target_density)(x)

    def terminal_cost_one(x):
        return log_ref_density(x) - log_target_density(x)

    # ----------------------------------------------------------
    # Gradient methods
    # ----------------------------------------------------------
    def make_pathwise_fns(n_steps):
        ts, means_by_step, chols_by_step, log_dets_by_step = prepare_discretization(n_steps)

        def loss_fn(params, key):
            key, ki, kne = jr.split(key, 3)
            x0 = sample_initial(ki, N_PARTICLES)
            nks = jr.split(kne, n_steps)

            def step(x, inp):
                t_n, t_np1, nk, means_s, chols_s, log_dets_s = inp
                dt = t_np1 - t_n
                eps = jr.normal(nk, x.shape)
                x_new, f_vals = _transition_batch_with_factors(
                    x, eps, t_n, t_np1, params, means_s, chols_s, log_dets_s
                )
                step_loss = dt / 2.0 * jnp.mean(jnp.sum(SIGMA2 * f_vals ** 2, axis=-1))
                return x_new, step_loss

            xT, step_losses = jax.lax.scan(
                step,
                x0,
                (ts[:-1], ts[1:], nks, means_by_step, chols_by_step, log_dets_by_step),
            )
            terminal_loss = jnp.mean(terminal_cost_batch(xT))
            return jnp.sum(step_losses) + terminal_loss

        @jit
        def loss_and_grad(params, key):
            loss, grads = jax.value_and_grad(loss_fn)(params, key)
            return loss, 0.0, grads

        return loss_and_grad

    def make_stopgrad_fns(n_steps):
        ts, means_by_step, chols_by_step, log_dets_by_step = prepare_discretization(n_steps)

        def loss_fn(params, params_sg, key):
            key, ki, kne = jr.split(key, 3)
            x0 = sample_initial(ki, N_PARTICLES)
            nks = jr.split(kne, n_steps)

            def step(x, inp):
                t_n, t_np1, nk, means_s, chols_s, log_dets_s = inp
                eps = jr.normal(nk, x.shape)
                x_new, _ = _transition_batch_with_factors(
                    x, eps, t_n, t_np1, params_sg, means_s, chols_s, log_dets_s
                )
                x_sg = jax.lax.stop_gradient(x)
                step_losses = vmap(step_cost_one, (0, None, None, None))(
                    x_sg, t_n, t_np1, params
                )
                return jax.lax.stop_gradient(x_new), jnp.mean(step_losses)

            xT, step_losses = jax.lax.scan(
                step,
                x0,
                (ts[:-1], ts[1:], nks, means_by_step, chols_by_step, log_dets_by_step),
            )
            terminal_loss = jnp.mean(terminal_cost_batch(xT))
            return jnp.sum(step_losses) + terminal_loss

        @jit
        def loss_and_grad(params, key):
            params_sg = jax.lax.stop_gradient(params)
            loss, grads = jax.value_and_grad(loss_fn)(params, params_sg, key)
            return loss, 0.0, grads

        return loss_and_grad

    def make_reinforce_fns(n_steps):
        ts, means_by_step, chols_by_step, log_dets_by_step = prepare_discretization(n_steps)

        @jit
        def loss_and_grad(params, key):
            key, ki, kne = jr.split(key, 3)
            x0 = sample_initial(ki, N_PARTICLES)
            nks = jr.split(kne, n_steps)
            score_init = jax.tree.map(
                lambda p: jnp.zeros((N_PARTICLES,) + p.shape, dtype=p.dtype), params
            )
            grad_init = jax.tree.map(jnp.zeros_like, params)

            def scan_step(carry, inp):
                x, score_tree, grad_acc, loss_acc = carry
                t_n, t_np1, nk, means_s, chols_s, log_dets_s = inp
                dt = t_np1 - t_n

                step_losses = vmap(step_cost_one, (0, None, None, None))(x, t_n, t_np1, params)
                step_grads = vmap(grad(step_cost_one, 3), (0, None, None, None))(
                    x, t_n, t_np1, params
                )
                step_g = tree_add(
                    tree_weighted_mean(step_losses, score_tree),
                    tree_mean(step_grads),
                )

                eps = jr.normal(nk, x.shape)
                x_new, _ = _transition_batch_with_factors(
                    x, eps, t_n, t_np1, params, means_s, chols_s, log_dets_s
                )

                def vjp_one(xi, ei):
                    _, vf = jax.vjp(lambda p: score_correction_fn(xi, t_n, p), params)
                    return vf(SIGMA * jnp.sqrt(dt) * ei)[0]

                score_inc = vmap(vjp_one)(x, eps)
                score_new = tree_add(score_tree, score_inc)
                grad_new = tree_add(grad_acc, step_g)
                loss_new = loss_acc + jnp.mean(step_losses)
                return (x_new, score_new, grad_new, loss_new), None

            (xT, score_T, grad_T, loss_T), _ = jax.lax.scan(
                scan_step,
                (x0, score_init, grad_init, 0.0),
                (ts[:-1], ts[1:], nks, means_by_step, chols_by_step, log_dets_by_step),
            )
            terminal_losses = terminal_cost_batch(xT)
            terminal_grad = tree_weighted_mean(terminal_losses, score_T)
            total_grad = tree_add(grad_T, terminal_grad)
            total_loss = loss_T + jnp.mean(terminal_losses)
            return total_loss, 0.0, total_grad

        return loss_and_grad

    # ----------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------
    def generate_base_samples(key, n_particles=2048, n_steps=100):
        ts, means_by_step, chols_by_step, log_dets_by_step = prepare_discretization(n_steps)
        key, ki, kne = jr.split(key, 3)
        x0 = sample_initial(ki, n_particles)
        nks = jr.split(kne, n_steps)

        def step(x, inp):
            t_n, t_np1, nk, means_s, chols_s, log_dets_s = inp
            dt = t_np1 - t_n
            eps = jr.normal(nk, x.shape)
            _, base_scores = _gmm_log_density_and_score_batch(
                x, means_s, chols_s, log_dets_s, log_weights, D
            )
            drift = 0.5 * x + SIGMA2 * base_scores
            x_new = x + drift * dt + SIGMA * jnp.sqrt(dt) * eps
            return x_new, None

        xT, _ = jax.lax.scan(
            step,
            x0,
            (ts[:-1], ts[1:], nks, means_by_step, chols_by_step, log_dets_by_step),
        )
        return xT

    def generate_samples(params, key, n_particles=2048, n_steps=100):
        ts, means_by_step, chols_by_step, log_dets_by_step = prepare_discretization(n_steps)
        key, ki, kne = jr.split(key, 3)
        x0 = sample_initial(ki, n_particles)
        nks = jr.split(kne, n_steps)

        def step(carry, inp):
            x, log_ratio = carry
            t_n, t_np1, nk, means_s, chols_s, log_dets_s = inp
            dt = t_np1 - t_n
            eps = jr.normal(nk, x.shape)
            x_new, f_vals = _transition_batch_with_factors(
                x, eps, t_n, t_np1, params, means_s, chols_s, log_dets_s
            )
            quad = dt / 2.0 * jnp.sum(SIGMA2 * f_vals ** 2, axis=-1)
            noise = jnp.sqrt(dt) * jnp.sum(SIGMA * f_vals * eps, axis=-1)
            return (x_new, log_ratio + quad + noise), None

        (xT, log_ratio), _ = jax.lax.scan(
            step,
            (x0, jnp.zeros(n_particles)),
            (ts[:-1], ts[1:], nks, means_by_step, chols_by_step, log_dets_by_step),
        )
        log_w = vmap(log_target_density)(xT) - log_ref_density_batch(xT) - log_ratio
        return xT, log_w

    # ----------------------------------------------------------
    # Return public API
    # ----------------------------------------------------------
    return types.SimpleNamespace(
        init_params=init_params,
        make_pathwise_fns=make_pathwise_fns,
        make_stopgrad_fns=make_stopgrad_fns,
        make_reinforce_fns=make_reinforce_fns,
        generate_base_samples=generate_base_samples,
        generate_samples=generate_samples,
        sigma=SIGMA,
        log_ref_density=log_ref_density,
        log_ou_marginal=log_ou_marginal,
        ou_marginal_score=ou_marginal_score,
        reverse_ou_drift=reverse_ou_drift,
        score_correction_fn=score_correction_fn,
        transition_batch=transition_batch,
        terminal_cost_one=terminal_cost_one,
    )


make_reverse_ou_dds_sampler = make_nonequilibrium_dds_sampler

