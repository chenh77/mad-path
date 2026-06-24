from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap


class BayesianGMMTarget:
    """Posterior target for a Bayesian Gaussian Mixture Model.

    Parameters
    ----------
    data : array-like, shape (n,) or (n, p)
        Observed data (converted to float32 JAX array; 1-D data are reshaped
        to (n, 1) so the code is uniform for all p).
    K : int
        Number of mixture components.
    m_0 : array-like, shape (p,), optional
        Prior mean for mu_k.  Defaults to the empirical mean.
    V_0 : array-like, shape (p, p), optional
        Prior covariance for mu_k.  Defaults to 9 * diag(var(data)).
    nu_0 : float, optional
        Degrees of freedom for the Wishart prior on each Sigma_k^{-1}.
        Must satisfy nu_0 > p - 1.  Defaults to p + 2.
    W_0 : array-like, shape (p, p), optional
        Scale matrix for the Wishart prior.  Defaults to diag(var(data)).
    alpha_0 : float, optional
        Symmetric Dirichlet concentration parameter.  Default 1 (uniform).
    ref_std : float, optional
        Standard deviation of the isotropic Gaussian reference distribution
        used by the DDS sampler.  Default 3.
    """

    def __init__(
        self,
        data,
        K: int = 2,
        m_0=None,
        V_0=None,
        nu_0: float | None = None,
        W_0=None,
        alpha_0: float = 1.0,
        ref_std: float = 3.0,
    ):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

        n, p = data.shape
        pL = p * (p + 1) // 2  # Cholesky entries per component

        self.K: int = K
        self.p: int = p
        self.n: int = n
        self.pL: int = pL
        self.dim: int = K * p + K * pL + (K - 1)
        self.ref_std: float = float(ref_std)
        self.alpha_0: float = float(alpha_0)

        # Store data as JAX array (treated as constant inside jit)
        self.data = jnp.array(data)

        # ------ default priors from data statistics ------
        data_mean = np.mean(data, axis=0)          # (p,)
        data_var = np.var(data, axis=0)            # (p,)

        self.m_0 = jnp.array(m_0 if m_0 is not None else data_mean, dtype=jnp.float32)
        self.V_0 = jnp.array(
            V_0 if V_0 is not None else np.diag(9.0 * data_var), dtype=jnp.float32
        )
        self.nu_0: float = float(nu_0 if nu_0 is not None else p + 2)
        self.W_0 = jnp.array(
            W_0 if W_0 is not None else np.diag(data_var), dtype=jnp.float32
        )

        # Precomputed inverses
        self._V0_inv = jnp.linalg.inv(self.V_0)
        self._W0_inv = jnp.linalg.inv(self.W_0)

        # Indices for assembling the lower-triangular Cholesky from a flat vector
        rows, cols = np.tril_indices(p)
        self._tril_rows = rows          # (pL,) numpy int arrays
        self._tril_cols = cols
        # Positions of diagonal elements within the flat tril vector
        self._diag_pos = np.where(rows == cols)[0]   # (p,) positions

        # Coefficient for each diagonal entry in the combined Wishart+Jacobian term:
        # sum_j (nu_0 - j) * l_diag[j]
        self._wishart_diag_coeff = jnp.array(
            [self.nu_0 - j for j in range(p)], dtype=jnp.float32
        )

    # ------------------------------------------------------------------
    # Low-level parameterization helpers
    # ------------------------------------------------------------------

    def _vec_to_L(self, v):
        """Flat vector (pL,) → lower-triangular Cholesky (p, p).

        Diagonal entries are stored as log(L[j,j]); off-diagonals as-is.
        """
        p = self.p
        L = jnp.zeros((p, p))
        L = L.at[self._tril_rows, self._tril_cols].set(v)
        # Exponentiate the diagonal: L[j,j] <- exp(stored value).
        # Clamp the exponent to [-10, 10] so L[j,j] stays in [exp(-10), exp(10)],
        # preventing float32 overflow and log(negative) NaN when samplers probe
        # extreme theta values.  The correction is exp(clip(v)) - v (subtract the
        # *original* raw_diag, not the clamped value) so L[j,j] = exp(clip(v)) > 0.
        raw_diag = jnp.diag(L)
        L = L + jnp.diag(jnp.exp(jnp.clip(raw_diag, -10.0, 10.0)) - raw_diag)
        return L

    def _parse_theta(self, theta):
        """Split flat theta into (mu, L_vecs, log_pi).

        Returns
        -------
        mu     : (K, p)
        L_vecs : (K, pL)  – each row is the flat log-Cholesky for one component
        log_pi : (K,)     – normalised log mixture weights
        """
        K, p, pL = self.K, self.p, self.pL
        mu = theta[: K * p].reshape(K, p)
        L_vecs = theta[K * p : K * p + K * pL].reshape(K, pL)
        free_logits = theta[K * p + K * pL :]          # (K-1,)
        logits = jnp.concatenate([free_logits, jnp.zeros(1)])
        log_pi = jax.nn.log_softmax(logits)
        return mu, L_vecs, log_pi

    # ------------------------------------------------------------------
    # Log-density components
    # ------------------------------------------------------------------

    def _log_gaussian_density(self, y, mu_k, L_k):
        """log N(y ; mu_k, (L_k L_k^T)^{-1}).

        L_k is the Cholesky factor of the *precision* matrix:
            Prec_k = L_k L_k^T,   Sigma_k = Prec_k^{-1}.

        log N = -p/2 log(2π) + sum_j log L_k[j,j] - 1/2 ||L_k^T (y-mu)||^2
        """
        p = self.p
        diff = y - mu_k
        log_det_half = jnp.sum(jnp.log(jnp.diag(L_k)))   # sum log(L[j,j])
        quad = jnp.sum(jnp.dot(L_k.T, diff) ** 2)
        return -0.5 * p * jnp.log(2.0 * jnp.pi) + log_det_half - 0.5 * quad

    def _log_likelihood(self, mu, L_all, log_pi):
        """Sum_i log sum_k pi_k N(y_i ; mu_k, Sigma_k)."""
        # log_comps[i, k] = log N(y_i ; mu_k, Sigma_k)
        log_comps = vmap(          # over data points
            lambda y: vmap(        # over K components
                lambda mu_k, L_k: self._log_gaussian_density(y, mu_k, L_k)
            )(mu, L_all)
        )(self.data)               # shape (n, K)

        log_obs = jax.scipy.special.logsumexp(
            log_pi[None, :] + log_comps, axis=1
        )                          # (n,)
        return jnp.sum(log_obs)

    def _log_prior_mu(self, mu):
        """sum_k log N(mu_k ; m_0, V_0)."""
        def _lp_k(mu_k):
            diff = mu_k - self.m_0
            return -0.5 * diff @ self._V0_inv @ diff   # drop constants
        return jnp.sum(vmap(_lp_k)(mu))

    def _log_prior_prec_and_jac(self, L_vecs):
        """sum_k [log p_Wishart(L_k L_k^T) + log|Jacobian_k|].

        The combined coefficient on the j-th log-diagonal entry is (nu_0 - j),
        derived in the module docstring.
        """
        def _lp_k(l_k):
            L_k = self._vec_to_L(l_k)
            l_diag = l_k[self._diag_pos]          # (p,) stored log-diag values

            # Combined Wishart + Jacobian diagonal term
            diag_term = jnp.sum(self._wishart_diag_coeff * l_diag)

            # Trace term: -1/2 tr(W_0^{-1} L_k L_k^T)
            Prec_k = L_k @ L_k.T
            trace_term = -0.5 * jnp.trace(self._W0_inv @ Prec_k)

            return diag_term + trace_term

        return jnp.sum(vmap(_lp_k)(L_vecs))

    def _log_prior_pi_and_jac(self, log_pi):
        """log p_Dirichlet(pi) + log|Jacobian_softmax|.

        With softmax parameterisation the combined term is alpha_0 * sum_k log(pi_k).
        """
        return self.alpha_0 * jnp.sum(log_pi)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_prob(self, theta):
        """Log unnormalized posterior density in unconstrained *theta* space."""
        mu, L_vecs, log_pi = self._parse_theta(theta)
        L_all = vmap(self._vec_to_L)(L_vecs)

        return (
            self._log_likelihood(mu, L_all, log_pi)
            + self._log_prior_mu(mu)
            + self._log_prior_prec_and_jac(L_vecs)
            + self._log_prior_pi_and_jac(log_pi)
        )

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
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def theta_to_params(self, theta):
        """Convert a single *theta* vector to interpretable parameters.

        Returns
        -------
        mu    : (K, p) component means
        Sigma : (K, p, p) component covariances  (= (L L^T)^{-1})
        pi    : (K,) mixture weights
        """
        mu, L_vecs, log_pi = self._parse_theta(theta)
        L_all = vmap(self._vec_to_L)(L_vecs)

        def _cov_from_L(L_k):
            L_inv = jnp.linalg.inv(L_k)
            return L_inv.T @ L_inv

        Sigma_all = vmap(_cov_from_L)(L_all)
        pi = jnp.exp(log_pi)
        return mu, Sigma_all, pi

    def init_theta(self, key, data_init: bool = True):
        """Return a reasonable starting point for theta.

        If *data_init* is True, initialises means near random data points
        (k-means style); otherwise draws from the prior.
        """
        K, p, pL = self.K, self.p, self.pL
        k1, k2, k3 = jr.split(key, 3)

        if data_init:
            # Pick K random data points as initial means (with small noise)
            idx = jr.choice(k1, self.n, shape=(K,), replace=False)
            mu_init = self.data[idx] + 0.1 * jr.normal(k2, (K, p))
        else:
            mu_init = self.m_0[None, :] + jr.normal(k1, (K, p))

        # Start with identity precision: L = I, so log-diag = 0, off-diag = 0
        L_init = jnp.zeros((K, pL))

        # Uniform mixture weights
        logit_init = jnp.zeros(K - 1)

        return jnp.concatenate([mu_init.ravel(), L_init.ravel(), logit_init])
