"""Mode discovery utilities for nonequilibrium DDS initialization."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax, vmap
from sklearn.cluster import KMeans

Array = jnp.ndarray


@dataclass(frozen=True)
class ModeInitializationConfig:
    """Tuning parameters for Langevin-based mode initialization."""

    num_particles: int = 500
    num_langevin_steps: int = 1000
    num_langevin_steps_per_beta: int | None = None
    initial_beta: float = 1e-3
    beta_schedule: Sequence[float] | None = None
    use_reference_score: bool = False
    beta_step_size_offset: float = 1e-8
    target_num_modes: int | None = None
    grad_clip: float | None = None
    mode_gd_steps: int = 20
    mode_gd_step_size: float = 1e-2
    mode_grad_clip: float | None = 10.0
    merge_radius_multiplier: float = 10.0
    merge_radius_floor: float = 1e-6
    merge_radius_recent_steps: int = 5
    merge_radius_use_fixed_stepsize: bool = False
    min_mode_count: int = 10
    covariance_floor: float = 1e-6
    kmeans_max_iter: int = 100
    kmeans_tol: float = 1e-4


@dataclass(frozen=True)
class ModeInitializationResult:
    """Result of Langevin-based mode initialization."""

    modes: Array
    covariances_diag: Array
    weights: Array
    labels: Array
    num_target_grad_queries: int
    initial_langevin_samples: Array
    langevin_samples: Array
    cluster_langevin_samples: Array
    mode_endpoints: Array
    merge_radius: float
    mode_counts: Array
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _clip_grad(x: Array, clip: float | None) -> Array:
    if clip is None:
        return x
    return jnp.clip(x, -float(clip), float(clip))


def _geometric_schedule(start: float, stop: float, num_steps: int) -> Array:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    start = _validate_positive("schedule start", start)
    stop = _validate_positive("schedule stop", stop)
    if num_steps == 1:
        return jnp.asarray([start], dtype=jnp.float32)
    exponent = jnp.linspace(0.0, 1.0, num_steps, dtype=jnp.float32)
    return start * (stop / start) ** exponent


def _validate_beta_schedule(beta_schedule: Sequence[float] | Array) -> Array:
    betas = jnp.asarray(beta_schedule, dtype=jnp.float32).reshape(-1)
    if betas.shape[0] == 0:
        raise ValueError("beta_schedule must contain at least one value.")
    if bool(jnp.any(~jnp.isfinite(betas))):
        raise ValueError("beta_schedule values must be finite.")
    if bool(jnp.any(betas <= 0.0)) or bool(jnp.any(betas > 1.0)):
        raise ValueError("beta_schedule values must lie in (0, 1].")
    return betas


def _default_beta_schedule(initial_beta: float, num_beta_levels: int) -> Array:
    initial_beta = _validate_positive("initial_beta", initial_beta)
    if num_beta_levels <= 0:
        raise ValueError("num_beta_levels must be positive.")
    if num_beta_levels == 1 or initial_beta >= 1.0:
        return jnp.ones((num_beta_levels,), dtype=jnp.float32)
    return _geometric_schedule(initial_beta, 1.0, num_beta_levels)


def _fallback_ref_score(ref_std: float) -> Callable[[Array], Array]:
    ref_var = float(ref_std) ** 2

    def score(x: Array) -> Array:
        return -x / ref_var

    return score


def _sample_reference(target: Any, key: Array, num_particles: int, dim: int) -> Array:
    if hasattr(target, "sample_ref"):
        return jnp.asarray(target.sample_ref(key, num_particles), dtype=jnp.float32)
    if not hasattr(target, "ref_std"):
        raise ValueError("target must define sample_ref(key, n) or ref_std.")
    return float(target.ref_std) * jr.normal(key, (num_particles, dim), dtype=jnp.float32)


def _run_langevin(
    target: Any,
    langevin_step_size: float,
    beta_schedule: Array,
    num_langevin_steps_per_beta: int,
    config: ModeInitializationConfig,
    rng_key: Array,
) -> tuple[Array, Array, Array, Array]:
    dim = int(target.dim)
    scan_key, init_key = jr.split(rng_key)
    x0 = _sample_reference(target, init_key, config.num_particles, dim)
    repeated_betas = jnp.repeat(
        beta_schedule,
        int(num_langevin_steps_per_beta),
    )
    keys = jr.split(scan_key, repeated_betas.shape[0])
    step_sizes = (
        jnp.asarray(langevin_step_size, dtype=jnp.float32)
        / (repeated_betas + jnp.asarray(config.beta_step_size_offset, dtype=jnp.float32))
    )

    target_score = vmap(jax.grad(target.log_prob))
    if hasattr(target, "log_ref"):
        ref_score = vmap(jax.grad(target.log_ref))
    else:
        ref_score = vmap(_fallback_ref_score(target.ref_std))

    def step(x: Array, inp: tuple[Array, Array, Array]) -> tuple[Array, None]:
        beta, h, key = inp
        tgt = _clip_grad(target_score(x), config.grad_clip)
        if config.use_reference_score:
            score = (1.0 - beta) * ref_score(x) + beta * tgt
        else:
            score = beta * tgt
        noise = jr.normal(key, x.shape, dtype=x.dtype)
        x_next = x + h * score + jnp.sqrt(2.0 * h) * noise
        return x_next, None

    x_final, _ = lax.scan(step, x0, (repeated_betas, step_sizes, keys))
    return x0, x_final, step_sizes, repeated_betas


def _ascend_to_modes(
    target: Any,
    samples: Array,
    config: ModeInitializationConfig,
) -> tuple[Array, Array]:
    target_score = vmap(jax.grad(target.log_prob))

    def step(x: Array, _: None) -> tuple[Array, Array]:
        score = _clip_grad(target_score(x), config.mode_grad_clip)
        displacement = config.mode_gd_step_size * score
        x_next = x + displacement
        displacement_norm = jnp.linalg.norm(displacement, axis=1)
        return x_next, displacement_norm

    endpoints, displacement_norms = lax.scan(
        step,
        samples,
        None,
        length=config.mode_gd_steps,
    )
    return endpoints, displacement_norms


def _automatic_merge_radius(
    displacement_norms: np.ndarray,
    config: ModeInitializationConfig,
    langevin_step_size: float = 0.0,
) -> float:
    if config.merge_radius_use_fixed_stepsize:
        return max(
            float(config.merge_radius_floor),
            float(config.merge_radius_multiplier) * float(langevin_step_size),
        )
    if displacement_norms.size == 0:
        return float(config.merge_radius_floor)
    recent_steps = min(config.merge_radius_recent_steps, displacement_norms.shape[0])
    recent = displacement_norms[-recent_steps:]
    finite_recent = recent[np.isfinite(recent)]
    if finite_recent.size == 0:
        scale = 0.0
    else:
        scale = float(np.median(finite_recent))
    return max(
        float(config.merge_radius_floor),
        float(config.merge_radius_multiplier) * scale,
    )


def _merge_endpoints_by_radius(
    endpoints: np.ndarray,
    logps: np.ndarray,
    merge_radius: float,
    min_mode_count: int,
    target_num_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily merge endpoints by Euclidean distance."""
    if endpoints.ndim != 2:
        raise ValueError("endpoints must have shape (N, D).")
    if endpoints.shape[0] == 0:
        return endpoints, np.zeros((0,), dtype=np.int32)

    order = np.argsort(-logps)
    centers: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for idx in order:
        x = endpoints[idx]
        if not centers:
            centers.append(x.copy())
            sums.append(x.copy())
            counts.append(1)
            continue

        distances = np.linalg.norm(np.stack(centers) - x[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= merge_radius:
            sums[nearest] = sums[nearest] + x
            counts[nearest] += 1
            centers[nearest] = sums[nearest] / counts[nearest]
        else:
            centers.append(x.copy())
            sums.append(x.copy())
            counts.append(1)

    kept = [
        (center, count)
        for center, count in zip(centers, counts)
        if count >= min_mode_count
    ]
    if not kept:
        return np.zeros((0, endpoints.shape[1]), dtype=endpoints.dtype), np.zeros(
            (0,), dtype=np.int32
        )

    kept_centers = np.stack([item[0] for item in kept])
    kept_counts = np.asarray([item[1] for item in kept], dtype=np.int32)
    order_by_count = np.argsort(-kept_counts)
    if target_num_modes is not None:
        order_by_count = order_by_count[: int(target_num_modes)]
    kept_centers = kept_centers[order_by_count]
    kept_counts = kept_counts[order_by_count]
    return kept_centers, kept_counts


def _fit_kmeans_cluster_stats(
    samples: np.ndarray,
    initial_centers: np.ndarray,
    covariance_floor: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, KMeans]:
    """Fit sklearn KMeans and compute diagonal variances and weights."""
    if initial_centers.ndim != 2:
        raise ValueError("initial_centers must have shape (K, D).")
    if initial_centers.shape[0] == 0:
        raise ValueError("initial_centers must contain at least one center.")

    k_fit = int(initial_centers.shape[0])
    km = KMeans(
        n_clusters=k_fit,
        init=initial_centers,
        n_init=1,
        max_iter=max_iter,
        tol=tol,
    )
    km.fit(samples)

    labels = km.labels_.astype(np.int32)
    var_diags = []
    weights = []
    counts = []
    for k in range(k_fit):
        samples_in_cluster = samples[labels == k]
        counts.append(samples_in_cluster.shape[0])
        if samples_in_cluster.shape[0] == 0:
            var_in_cluster = np.full(
                (samples.shape[1],), covariance_floor, dtype=samples.dtype
            )
        else:
            var_in_cluster = np.asarray(
                jnp.var(jnp.asarray(samples_in_cluster), axis=0)
            )
            var_in_cluster = np.maximum(var_in_cluster, covariance_floor)
        var_diags.append(var_in_cluster)
        weights.append(samples_in_cluster.shape[0] / samples.shape[0])

    return (
        np.asarray(km.cluster_centers_),
        np.asarray(var_diags),
        np.asarray(weights, dtype=np.float32),
        labels,
        np.asarray(counts, dtype=np.int32),
        km,
    )


def initialize_modes_from_langevin(
    target: Any,
    langevin_step_size: float | None = None,
    beta_schedule: Sequence[float] | Array | None = None,
    *,
    initial_langevin_stepsize: float | None = None,
    terminal_langevin_stepsize: float | None = None,
    num_particles: int | None = None,
    num_langevin_steps: int | None = None,
    num_langevin_steps_per_beta: int | None = None,
    initial_beta: float | None = None,
    target_num_modes: int | None = None,
    rng_key: Array | None = None,
    config: ModeInitializationConfig | None = None,
    **config_overrides: Any,
) -> ModeInitializationResult:
    """Discover modes and diagonal covariances from annealed Langevin samples.

    Args:
        target: Object exposing ``log_prob(x)``, ``dim``, and preferably
            ``log_ref(x)`` and ``sample_ref(key, n)``.
        langevin_step_size: Fixed ULA time step ``h`` used at each beta.
        beta_schedule: Inverse temperatures. Each value is repeated for
            ``num_langevin_steps_per_beta`` ULA steps.
            By default the Langevin score is ``beta * grad target``, matching
            the convention used in this repository's experiment scripts. Set
            ``config.use_reference_score=True`` to use the geometric bridge
            score ``(1 - beta) * grad ref + beta * grad target``.
            The actual ULA time step is
            ``langevin_step_size / (beta + config.beta_step_size_offset)``.
        initial_langevin_stepsize: Backward-compatible alias for
            ``langevin_step_size``.
        terminal_langevin_stepsize: Accepted for backward compatibility; fixed
            step-size Langevin uses ``langevin_step_size``.
        num_particles: Optional override for the number of Langevin particles.
        num_langevin_steps: Backward-compatible alias for the number of ULA
            steps per beta when ``num_langevin_steps_per_beta`` is not set.
        num_langevin_steps_per_beta: Number of ULA steps to run at each beta.
        initial_beta: Used only to build a default beta schedule when
            ``beta_schedule`` is not provided.
        target_num_modes: Optional cap on retained modes by mode count.
        rng_key: JAX PRNG key. Defaults to ``jr.PRNGKey(0)``.
        config: Optional full configuration object.
        **config_overrides: Additional ``ModeInitializationConfig`` field
            overrides.

    Returns:
        ``ModeInitializationResult`` with GMM-style parameters.
    """
    if not hasattr(target, "log_prob"):
        raise ValueError("target must define log_prob(x).")
    if not hasattr(target, "dim"):
        raise ValueError("target must define dim.")

    if config is None:
        config = ModeInitializationConfig()
    override_values = dict(config_overrides)
    if num_particles is not None:
        override_values["num_particles"] = num_particles
    if num_langevin_steps is not None:
        override_values["num_langevin_steps"] = num_langevin_steps
    if num_langevin_steps_per_beta is not None:
        override_values["num_langevin_steps_per_beta"] = num_langevin_steps_per_beta
    if initial_beta is not None:
        override_values["initial_beta"] = initial_beta
    if beta_schedule is not None:
        override_values["beta_schedule"] = beta_schedule
    if target_num_modes is not None:
        override_values["target_num_modes"] = target_num_modes
    if override_values:
        config = ModeInitializationConfig(
            **{**config.__dict__, **override_values}
        )

    if config.num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if config.num_langevin_steps <= 0:
        raise ValueError("num_langevin_steps must be positive.")
    if (
        config.num_langevin_steps_per_beta is not None
        and config.num_langevin_steps_per_beta <= 0
    ):
        raise ValueError("num_langevin_steps_per_beta must be positive.")
    if config.mode_gd_steps <= 0:
        raise ValueError("mode_gd_steps must be positive.")
    if config.mode_gd_step_size <= 0.0:
        raise ValueError("mode_gd_step_size must be positive.")
    if config.min_mode_count <= 0:
        raise ValueError("min_mode_count must be positive.")
    if config.target_num_modes is not None and config.target_num_modes <= 0:
        raise ValueError("target_num_modes must be positive when provided.")
    if config.kmeans_max_iter <= 0:
        raise ValueError("kmeans_max_iter must be positive.")
    if config.kmeans_tol < 0.0:
        raise ValueError("kmeans_tol must be nonnegative.")
    if config.beta_step_size_offset < 0.0:
        raise ValueError("beta_step_size_offset must be nonnegative.")

    if langevin_step_size is None:
        if initial_langevin_stepsize is None:
            raise ValueError(
                "langevin_step_size is required unless initial_langevin_stepsize "
                "is provided."
            )
        langevin_step_size = initial_langevin_stepsize
    langevin_step_size = _validate_positive("langevin_step_size", langevin_step_size)
    if terminal_langevin_stepsize is not None:
        _validate_positive("terminal_langevin_stepsize", terminal_langevin_stepsize)

    if config.beta_schedule is None:
        beta_levels = _default_beta_schedule(config.initial_beta, config.num_langevin_steps)
        steps_per_beta = 1
    else:
        beta_levels = _validate_beta_schedule(config.beta_schedule)
        steps_per_beta = (
            config.num_langevin_steps_per_beta
            if config.num_langevin_steps_per_beta is not None
            else config.num_langevin_steps
        )

    if rng_key is None:
        rng_key = jr.PRNGKey(0)

    initial_langevin_samples, langevin_samples, step_sizes, betas = _run_langevin(
        target,
        langevin_step_size,
        beta_levels,
        steps_per_beta,
        config,
        rng_key,
    )
    samples_np = np.asarray(langevin_samples)
    finite_samples = np.all(np.isfinite(samples_np), axis=1)
    valid_samples_np = samples_np[finite_samples]
    finite_langevin_samples_np = valid_samples_np.copy()
    num_finite_langevin_samples = int(valid_samples_np.shape[0])
    if valid_samples_np.shape[0] == 0:
        raise RuntimeError("No finite Langevin samples were produced.")

    valid_samples = jnp.asarray(valid_samples_np, dtype=jnp.float32)
    mode_endpoints, displacement_norms = _ascend_to_modes(target, valid_samples, config)
    endpoints_np = np.asarray(mode_endpoints)
    logps_np = np.asarray(vmap(target.log_prob)(mode_endpoints))
    finite_endpoints = np.isfinite(logps_np) & np.all(np.isfinite(endpoints_np), axis=1)
    endpoints_np = endpoints_np[finite_endpoints]
    logps_np = logps_np[finite_endpoints]
    valid_samples_np = valid_samples_np[finite_endpoints]

    if endpoints_np.shape[0] == 0:
        raise RuntimeError("No finite mode endpoints were produced.")

    merge_radius = _automatic_merge_radius(np.asarray(displacement_norms), config, langevin_step_size)
    proto_centers, proto_counts = _merge_endpoints_by_radius(
        endpoints_np,
        logps_np,
        merge_radius,
        config.min_mode_count,
        config.target_num_modes,
    )
    if proto_centers.shape[0] == 0:
        raise RuntimeError(
            "No modes survived merging. Try reducing min_mode_count or merge radius."
        )

    modes_np, covs_np, weights_np, labels_np, assigned_counts, km = (
        _fit_kmeans_cluster_stats(
            valid_samples_np,
            proto_centers,
            config.covariance_floor,
            config.kmeans_max_iter,
            config.kmeans_tol,
        )
    )
    nonempty = assigned_counts > 0
    if not np.all(nonempty):
        modes_np, covs_np, weights_np, labels_np, assigned_counts, km = (
            _fit_kmeans_cluster_stats(
                valid_samples_np,
                modes_np[nonempty],
                config.covariance_floor,
                config.kmeans_max_iter,
                config.kmeans_tol,
            )
        )
    assigned_counts = np.bincount(labels_np, minlength=modes_np.shape[0]).astype(
        np.int32
    )
    num_target_grad_queries = (
        config.num_particles * int(betas.shape[0])
        + num_finite_langevin_samples * config.mode_gd_steps
    )

    diagnostics = {
        "num_particles": config.num_particles,
        "num_beta_levels": int(beta_levels.shape[0]),
        "num_langevin_steps_per_beta": int(steps_per_beta),
        "num_langevin_steps": int(betas.shape[0]),
        "num_valid_langevin_samples": num_finite_langevin_samples,
        "num_samples_used_for_covariance": int(valid_samples_np.shape[0]),
        "num_finite_mode_endpoints": int(endpoints_np.shape[0]),
        "num_dropped_mode_endpoints": int(
            num_finite_langevin_samples - valid_samples_np.shape[0]
        ),
        "langevin_step_size": float(langevin_step_size),
        "initial_actual_langevin_stepsize": float(step_sizes[0]),
        "terminal_actual_langevin_stepsize": float(step_sizes[-1]),
        "actual_langevin_stepsizes": step_sizes,
        "beta_step_size_offset": float(config.beta_step_size_offset),
        "initial_beta": float(betas[0]),
        "terminal_beta": float(betas[-1]),
        "beta_schedule": beta_levels,
        "proto_mode_counts": jnp.asarray(proto_counts),
        "assigned_mode_counts": jnp.asarray(assigned_counts),
        "kmeans_n_iter": int(km.n_iter_),
        "kmeans_initial_centers": jnp.asarray(proto_centers, dtype=jnp.float32),
    }

    return ModeInitializationResult(
        modes=jnp.asarray(modes_np, dtype=jnp.float32),
        covariances_diag=jnp.asarray(covs_np, dtype=jnp.float32),
        weights=jnp.asarray(weights_np, dtype=jnp.float32),
        labels=jnp.asarray(labels_np, dtype=jnp.int32),
        num_target_grad_queries=int(num_target_grad_queries),
        initial_langevin_samples=jnp.asarray(initial_langevin_samples, dtype=jnp.float32),
        langevin_samples=jnp.asarray(finite_langevin_samples_np, dtype=jnp.float32),
        cluster_langevin_samples=jnp.asarray(valid_samples_np, dtype=jnp.float32),
        mode_endpoints=jnp.asarray(endpoints_np, dtype=jnp.float32),
        merge_radius=float(merge_radius),
        mode_counts=jnp.asarray(assigned_counts, dtype=jnp.int32),
        diagnostics=diagnostics,
    )


__all__ = [
    "ModeInitializationConfig",
    "ModeInitializationResult",
    "initialize_modes_from_langevin",
]
