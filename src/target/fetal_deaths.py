from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp


# ---------------------------------------------------------------------------
# HS3 dataset (Table 1, Tjelmeland & Hegstad 2001)
# Each tuple: (litter_size n, dead_fetuses x, number_of_litters)
# Only nonzero counts are listed.
# ---------------------------------------------------------------------------
_HS3_DATA = [
    # n=1
    (1, 0, 7),
    # n=2
    (2, 0, 7),
    # n=3
    (3, 0, 6),
    # n=4
    (4, 0, 5), (4, 1, 2), (4, 2, 1),
    # n=5
    (5, 0, 8), (5, 1, 2), (5, 2, 1), (5, 4, 1), (5, 5, 1),
    # n=6
    (6, 0, 8),
    # n=7
    (7, 0, 4), (7, 1, 4), (7, 2, 2), (7, 3, 1),
    # n=8
    (8, 0, 7), (8, 1, 7), (8, 2, 1),
    # n=9
    (9, 0, 8), (9, 1, 9), (9, 2, 7), (9, 3, 1), (9, 4, 1),
    # n=10
    (10, 0, 22), (10, 1, 17), (10, 2, 2), (10, 4, 1), (10, 7, 1), (10, 8, 1),
    # n=11
    (11, 0, 30), (11, 1, 18), (11, 2, 9), (11, 3, 1), (11, 4, 2), (11, 6, 1), (11, 8, 1),
    # n=12
    (12, 0, 54), (12, 1, 27), (12, 2, 12), (12, 3, 2), (12, 4, 1), (12, 6, 2), (12, 7, 1),
    # n=13
    (13, 0, 46), (13, 1, 30), (13, 2, 8), (13, 3, 4), (13, 4, 1), (13, 5, 1), (13, 7, 1),
    # n=14
    (14, 0, 43), (14, 1, 21), (14, 2, 13), (14, 3, 3), (14, 4, 1), (14, 7, 1), (14, 9, 1),
    # n=15
    (15, 0, 22), (15, 1, 22), (15, 2, 5), (15, 3, 2), (15, 4, 1),
    # n=16
    (16, 0, 6), (16, 1, 6), (16, 2, 3), (16, 4, 1), (16, 5, 1),
    # n=17: all zero counts, omitted
    # n=18
    (18, 0, 3), (18, 2, 2), (18, 3, 1),
]


def _build_data_arrays():
    ns = np.array([row[0] for row in _HS3_DATA], dtype=np.float32)
    xs = np.array([row[1] for row in _HS3_DATA], dtype=np.float32)
    cts = np.array([row[2] for row in _HS3_DATA], dtype=np.float32)
    # precompute log C(n, x) = lgamma(n+1) - lgamma(x+1) - lgamma(n-x+1)
    from scipy.special import gammaln
    log_cnx = gammaln(ns + 1) - gammaln(xs + 1) - gammaln(ns - xs + 1)
    return (
        jnp.array(ns),
        jnp.array(xs),
        jnp.array(cts),
        jnp.array(log_cnx.astype(np.float32)),
    )


class FetalDeathsTarget:
    """Posterior over (γ̃, μ̃, θ̃, ν̃) for the fetal-deaths mixture model.

    dim = 4: unconstrained parameters (logit-γ, logit-μ, log-θ, logit-ν).

    Parameters
    ----------
    ref_std : float
        Std of the isotropic Gaussian reference used by DDS.  Default 1.0.
    """

    dim = 4
    K_modes = 2
    _PARAM_NAMES = ["γ", "μ", "θ", "ν"]   # constrained-space names (returned by theta_to_params)

    def __init__(self, ref_std: float = 1.0):
        self.ref_std = float(ref_std)
        self._ns, self._xs, self._counts, self._log_cnx = _build_data_arrays()

    # ------------------------------------------------------------------
    # Likelihood helpers
    # ------------------------------------------------------------------

    def _log_p_bb(self, mu: jax.Array, theta: jax.Array) -> jax.Array:
        """Log beta-binomial prob for every data row; shape (M,)."""
        a = mu / theta            # alpha = μ/θ
        b = (1.0 - mu) / theta    # beta  = (1-μ)/θ
        lp = (
            self._log_cnx
            + jsp.special.gammaln(self._xs + a)   - jsp.special.gammaln(a)
            + jsp.special.gammaln(self._ns - self._xs + b) - jsp.special.gammaln(b)
            - jsp.special.gammaln(self._ns + a + b) + jsp.special.gammaln(a + b)
        )
        return lp

    def _log_p_bin(self, nu: jax.Array) -> jax.Array:
        """Log binomial prob for every data row; shape (M,)."""
        return (
            self._log_cnx
            + self._xs * jnp.log(nu)
            + (self._ns - self._xs) * jnp.log(1.0 - nu)
        )

    # ------------------------------------------------------------------
    # Log-posterior in unconstrained space
    # ------------------------------------------------------------------

    def log_prob(self, theta: jax.Array) -> jax.Array:
        """Log-posterior at unconstrained θ = (logit_γ, logit_μ, log_θ, logit_ν).

        Includes log-Jacobians for the change of variables so that the
        target is a proper density on ℝ⁴.

        Parameters
        ----------
        theta : array, shape (4,)

        Returns
        -------
        scalar
        """
        gamma_t, mu_t, theta_t, nu_t = theta[0], theta[1], theta[2], theta[3]

        # back-transform
        gamma = jax.nn.sigmoid(gamma_t)
        mu    = jax.nn.sigmoid(mu_t)
        th    = jnp.exp(theta_t)          # θ > 0
        nu    = jax.nn.sigmoid(nu_t)

        # --- log-likelihood ---
        log_p_bb  = self._log_p_bb(mu, th)    # (M,)
        log_p_bin = self._log_p_bin(nu)        # (M,)

        log_mix = jsp.special.logsumexp(
            jnp.stack([
                jnp.log(gamma)       + log_p_bb,
                jnp.log(1.0 - gamma) + log_p_bin,
            ], axis=0),
            axis=0,
        )  # (M,)

        log_lik = jnp.dot(self._counts, log_mix)

        # --- log-prior + log-Jacobian (in unconstrained space) ---
        # Uniform(0,1) priors: log|J| = log(sigmoid(z̃)(1-sigmoid(z̃)))
        def _log_jac_logistic(t):
            # log(sigmoid(t)) + log(1 - sigmoid(t))  = t - 2*softplus(t)
            return t - 2.0 * jax.nn.softplus(t)

        log_jac_logistic = (
            _log_jac_logistic(gamma_t)
            + _log_jac_logistic(mu_t)
            + _log_jac_logistic(nu_t)
        )
        # Exponential(1) prior on θ = exp(θ̃):  log p(θ̃) = -exp(θ̃) + θ̃
        log_prior_theta = -th + theta_t

        return log_lik + log_jac_logistic + log_prior_theta

    # ------------------------------------------------------------------
    # Reference distribution  q₀ = N(0, ref_std² I)
    # ------------------------------------------------------------------

    def log_ref(self, theta: jax.Array) -> jax.Array:
        s2 = self.ref_std ** 2
        return -0.5 * (self.dim * jnp.log(2.0 * jnp.pi * s2) + jnp.dot(theta, theta) / s2)

    def sample_ref(self, key: jax.Array, n: int) -> jax.Array:
        return jax.random.normal(key, (n, self.dim)) * self.ref_std

    # ------------------------------------------------------------------
    # Convenience: map between constrained and unconstrained
    # ------------------------------------------------------------------

    def constrained(self, theta: jax.Array):
        """Return (γ, μ, θ, ν) from unconstrained θ̃."""
        gamma_t, mu_t, theta_t, nu_t = theta[0], theta[1], theta[2], theta[3]
        return (
            jax.nn.sigmoid(gamma_t),
            jax.nn.sigmoid(mu_t),
            jnp.exp(theta_t),
            jax.nn.sigmoid(nu_t),
        )

    def theta_to_params(self, theta: jax.Array):
        """Decode unconstrained θ̃ → constrained (γ, μ, θ, ν) for visualization.

        Returns (params, params, dummy), following the same uniform 3-tuple
        interface as the other target classes:
            params : shape (4,) — constrained parameters
            second : same (no curve/predictive in this model)
            third  : dummy scalar
        """
        gamma, mu, th, nu = self.constrained(theta)
        params = jnp.stack([gamma, mu, th, nu])
        return params, params, jnp.ones(1)


def make_fetal_deaths_target(ref_std: float = 1.0) -> FetalDeathsTarget:
    """Construct the FetalDeathsTarget with the HS3 dataset."""
    return FetalDeathsTarget(ref_std=ref_std)
