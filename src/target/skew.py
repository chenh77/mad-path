from typing import Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap
from jax.scipy.special import log_ndtr, logsumexp


# ============================================================
# Helper functions
# ============================================================
def _as_2d(x):
    """Convert x to (N, d) shape."""
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x


def _log_phi_std(z):
    """Log standard normal pdf: log φ(z) = -0.5*z^2 - 0.5*log(2π)."""
    return -0.5 * z * z - 0.5 * jnp.log(2.0 * jnp.pi)


def _score_from_logprob(log_prob_fn):
    """Return a score function from a log_prob function."""
    def score_fn(x):
        x_2d = _as_2d(x)
        def lp(x_1d):
            return log_prob_fn(x_1d[None, :])
        g = vmap(grad(lp))(x_2d)
        return g[0] if x.ndim == 1 else g
    return score_fn


# ============================================================
# SkewNormalMixtureIID (component-specific alpha_k)
# ============================================================
class SkewNormalMixtureIID:
    """Skew-normal mixture with component-specific skewness parameters.

    Parameters
    ----------
    mus : array, shape (K, d)
        Component means.
    sigmas : array, shape (K,)
        Component standard deviations (scalar per component, isotropic).
    weights : array, shape (K,), optional
        Mixture weights (will be normalized internally). Defaults to uniform.
    alpha : scalar or array, shape (K,), optional
        Skewness parameters (scalar or per-component). Defaults to 10.0.
    ref_std : float, optional
        Standard deviation of the isotropic Gaussian reference. Defaults to 1.0.
    """

    def __init__(
        self,
        mus,
        sigmas,
        weights: Optional[jnp.ndarray] = None,
        alpha: Union[float, jnp.ndarray] = 10.0,
        ref_std: float = 1.0,
    ):
        mus = jnp.asarray(mus, dtype=jnp.float32)
        if mus.ndim != 2:
            raise ValueError(f"mus must be (K,d), got {mus.shape}")
        self.mus = mus
        K, d = mus.shape
        self._K = int(K)  # private: not a mixture model in Bayesian sense
        self.d = int(d)
        self.dim: int = int(d)

        sigmas = jnp.asarray(sigmas, dtype=jnp.float32).reshape(-1)
        if sigmas.shape[0] != self._K:
            raise ValueError(f"sigmas must have length K={self._K}, got {sigmas.shape[0]}")
        if jnp.any(sigmas <= 0):
            raise ValueError("sigmas must be strictly positive")
        self.sigmas = sigmas

        if weights is None:
            self.weights = jnp.ones((self._K,), dtype=jnp.float32) / self._K
        else:
            w = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
            if w.shape[0] != self._K:
                raise ValueError(f"weights must have length K={self._K}, got {w.shape[0]}")
            if jnp.any(w <= 0):
                raise ValueError("weights must be strictly positive")
            self.weights = w / jnp.sum(w)

        a = jnp.asarray(alpha, dtype=jnp.float32).reshape(-1)
        if a.shape[0] == 1:
            a = jnp.repeat(a, self._K)
        if a.shape[0] != self._K:
            raise ValueError(f"alpha must be scalar or length K={self._K}, got {a.shape[0]}")
        self.alphas = a

        self.ref_std: float = float(ref_std)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def log_prob(self, x):
        """Log unnormalized target density at *x*.

        x : (N, d) or (d,) -> (N,) or scalar
        pi(x) ∝ Σ_k w_k Π_j (2/σ_k) φ(z) Φ(alpha_k z)
        with z = (x - mu_k) / sigma_k
        """
        x_orig = x
        x = _as_2d(x)
        if x.shape[1] != self.d:
            raise ValueError(f"x must have last dim {self.d}, got {x.shape}")

        # Broadcast to (N, K, d)
        x_ = x[:, None, :]
        mu = self.mus[None, :, :]
        sigma = self.sigmas[None, :, None]
        alpha = self.alphas[None, :, None]

        z = (x_ - mu) / sigma
        log_phi = _log_phi_std(z)
        log_Phi = log_ndtr(alpha * z)  # stable log Φ

        # sum over d, then include -d log sigma
        comp_lp = jnp.sum(log_phi + log_Phi, axis=-1) - self.d * jnp.log(sigma.squeeze(-1))
        wlog = jnp.log(self.weights)[None, :]
        lp = logsumexp(comp_lp + wlog, axis=-1)

        # Return scalar if input was 1D, otherwise return vector
        return lp[0] if x_orig.ndim == 1 else lp

    def log_ref(self, x):
        """Log isotropic Gaussian reference density at *x*."""
        ref_var = self.ref_std ** 2
        return (
            -0.5 * jnp.sum(x ** 2) / ref_var
            - 0.5 * self.dim * jnp.log(2.0 * jnp.pi * ref_var)
        )

    def sample_ref(self, key, n: int):
        """Draw *n* i.i.d. samples from the Gaussian reference."""
        return self.ref_std * jr.normal(key, (n, self.dim))

    # ------------------------------------------------------------------
    # Additional methods for diagnostics
    # ------------------------------------------------------------------
    def score(self, x):
        """Gradient of log_prob at x."""
        return grad(self.log_prob)(_as_2d(x))

    def sample(self, key: jr.PRNGKey, n: int) -> Tuple[jnp.ndarray, jr.PRNGKey]:
        """Draw *n* i.i.d. samples from the mixture.

        Per-component skew-normal sampling (Azzalini form):
        If alpha_k given, delta = alpha / sqrt(1 + alpha^2)
        y = delta*|z0| + sqrt(1 - delta^2)*z1
        x = mu + sigma*y
        """
        n = int(n)
        key, k_idx, k0, k1 = jr.split(key, 4)

        idx = jr.categorical(k_idx, jnp.log(self.weights), shape=(n,))  # (n,)
        mu = self.mus[idx, :]  # (n, d)
        sigma = self.sigmas[idx][:, None]  # (n, 1)

        alpha_k = self.alphas[idx][:, None]  # (n, 1)
        delta = alpha_k / jnp.sqrt(1.0 + alpha_k * alpha_k)
        s = jnp.sqrt(1.0 - delta * delta)

        z0 = jr.normal(k0, (n, self.d), dtype=jnp.float32)
        z1 = jr.normal(k1, (n, self.d), dtype=jnp.float32)
        y = delta * jnp.abs(z0) + s * z1
        x = mu + sigma * y
        return x, key

    def true_mean(self) -> jnp.ndarray:
        """E[X] = Σ_k w_k (mu_k + sigma_k * delta_k * sqrt(2/pi))
        where delta_k = alpha_k / sqrt(1 + alpha_k^2).
        """
        delta = self.alphas / jnp.sqrt(1.0 + self.alphas ** 2)  # (K,)
        skew_offset = self.sigmas * delta * jnp.sqrt(2.0 / jnp.pi)  # (K,)
        component_means = self.mus + skew_offset[:, None]  # (K, d)
        return jnp.sum(self.weights[:, None] * component_means, axis=0)  # (d,)

    def true_cov(self) -> jnp.ndarray:
        """Cov[X] = Σ_k w_k (C_k + (mu_k* - mu*)(mu_k* - mu*)^T)
        where mu_k* = component mean,
        C_k = diag(sigma_k^2 * (1 - 2*delta_k^2/pi)) (marginal variance, diagonal).
        """
        delta = self.alphas / jnp.sqrt(1.0 + self.alphas ** 2)  # (K,)
        comp_var = self.sigmas ** 2 * (1.0 - 2.0 * delta ** 2 / jnp.pi)  # (K,)

        skew_offset = self.sigmas * delta * jnp.sqrt(2.0 / jnp.pi)  # (K,)
        comp_means = self.mus + skew_offset[:, None]  # (K, d)
        mix_mean = jnp.sum(self.weights[:, None] * comp_means, axis=0)  # (d,)

        diff = comp_means - mix_mean[None, :]  # (K, d)
        # Σ_k w_k * (diag(comp_var_k) + diff_k diff_k^T)
        cov = jnp.sum(
            self.weights[:, None, None]
            * (
                jnp.eye(self.d, dtype=jnp.float32)[None, :, :]
                * comp_var[:, None, None]
                + diff[:, :, None] * diff[:, None, :]
            ),
            axis=0,
        )
        return cov

    def theta_to_params(self, theta):
        """Identity mapping: skew mixture is already in unconstrained space.

        Returns (theta, None, None) — uniform 3-tuple interface shared across
        target classes; the skew mixture has no separate curve/predictive output.
        """
        return theta, None, None


# ============================================================
# Factory function
# ============================================================
def build_case1_mixture(
    *,
    d: int,
    mu_levels=(8.0, 20.0),
    sigma: float = 1.0,
    alpha: Union[float, jnp.ndarray] = 10.0,
    eps_tie: float = 1e-3,
    pair_jitter: float = 0.0,
    seed: int = 0,
    dtype=jnp.float32,
):
    """Build a 13-component structured skew-normal mixture.

    Structured centers in R^2, repeated across d/2 pairs to get R^d.

    Parameters
    ----------
    d : int
        Dimension (must be even).
    mu_levels : tuple of 2 floats
        (inner, outer) mean levels.
    sigma : float
        Component standard deviation.
    alpha : float or array
        Skewness parameter (scalar or per-component).
    eps_tie : float
        Stagger amount for pair coupling.
    pair_jitter : float
        Random jitter on mus.
    seed : int
        Random seed.
    dtype
        JAX dtype.

    Returns
    -------
    mus : (13, d)
    sigmas : (13,)
    weights : (13,)
    alphas : (13,)
    """
    if d % 2 != 0:
        raise ValueError(f"d must be even, got {d}")

    m = d // 2
    inner, outer = float(mu_levels[0]), float(mu_levels[1])

    mus_2d = jnp.array(
        [
            [0.0, 0.0],
            [inner, inner],
            [-inner, inner],
            [inner, -inner],
            [-inner, -inner],
            [outer, outer],
            [outer, 0.0],
            [outer, -outer],
            [0.0, outer],
            [0.0, -outer],
            [-outer, outer],
            [-outer, 0.0],
            [-outer, -outer],
        ],
        dtype=dtype,
    )  # (13, 2)

    K = mus_2d.shape[0]
    mus = jnp.tile(mus_2d[:, None, :], (1, m, 1))  # (13, m, 2)

    if eps_tie > 0:
        idx = jnp.arange(m, dtype=dtype)
        denom = max(1.0, float(m - 1))
        stagger = (float(eps_tie) * idx / denom)[:, None]  # (m, 1)
        pair_shift = jnp.concatenate([stagger, -stagger], axis=1)  # (m, 2)
        mus = mus + pair_shift[None, :, :]

    if pair_jitter > 0:
        key = jr.PRNGKey(seed)
        jit = float(pair_jitter) * jr.normal(key, (m, 2), dtype=dtype)
        mus = mus + jit[None, :, :]

    mus = mus.reshape(K, d)
    sigmas = jnp.full((K,), float(sigma), dtype=dtype)
    weights = jnp.full((K,), 1.0 / K, dtype=dtype)

    alpha_t = jnp.asarray(alpha, dtype=dtype).reshape(-1)
    if alpha_t.size == 1:
        alpha_t = jnp.repeat(alpha_t, K)
    if alpha_t.size != K:
        raise ValueError(f"alpha must be scalar or length K={K}, got {alpha_t.size}")

    return mus, sigmas, weights, alpha_t


def make_skew_mixture_target(
    *,
    d: int = 2,
    mu_levels=(8.0, 20.0),
    sigma: float = 1.0,
    alpha: Union[float, jnp.ndarray] = 10.0,
    eps_tie: float = 1e-3,
    pair_jitter: float = 0.0,
    seed: int = 0,
    ref_std: float = 1.0,
) -> SkewNormalMixtureIID:
    """Create a skew-normal mixture target from case1 parameters."""
    mus, sigmas, weights, alphas = build_case1_mixture(
        d=d,
        mu_levels=mu_levels,
        sigma=sigma,
        alpha=alpha,
        eps_tie=eps_tie,
        pair_jitter=pair_jitter,
        seed=seed,
        dtype=jnp.float32,
    )

    return SkewNormalMixtureIID(
        mus=mus,
        sigmas=sigmas,
        weights=weights,
        alpha=alphas,
        ref_std=ref_std,
    )
