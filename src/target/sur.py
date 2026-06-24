import jax.numpy as jnp
import numpy as np
from jax import grad


class SURTarget:
    """SUR profile likelihood target.

    Each response m is regressed on its own covariate set:
        y_m = X_m β_m + ε_m,   ε ~ MN(0, Σ ⊗ I_T)

    The profile likelihood integrates out Σ analytically.
    """

    def __init__(self, Y, X_list, ref_std=1.0):
        """
        Parameters
        ----------
        Y : array, shape (T, M)
            Response matrix.
        X_list : list of M arrays, each shape (T, k_m)
            Design matrices for each equation.
        ref_std : float
            Std of the isotropic Gaussian reference (used by DDS as q₀).
        """
        self.Y = jnp.asarray(Y, dtype=jnp.float32)
        self.X_list = [jnp.asarray(X, dtype=jnp.float32) for X in X_list]

        self.N = int(self.Y.shape[0])   # kept as N for backward compat; equals T
        self.M = int(self.Y.shape[1])
        self.J_list = [int(X.shape[1]) for X in self.X_list]
        self.dim = sum(self.J_list)
        self._splits = list(np.cumsum(self.J_list[:-1]))

        self.ref_std = float(ref_std)
        self.K_modes = 2
        self._known_modes = None
        self._PARAM_NAMES = [f'β{m+1}' for m in range(self.M)]

    def _log_lik(self, theta):
        """Profile log-likelihood. theta: (batch, dim) → (batch,)."""
        betas = jnp.split(theta, self._splits, axis=1)

        # r_m = y_m - X_m β_m,  shape (batch, T) each
        residuals = [
            self.Y[:, m][None, :] - jnp.dot(betas[m], self.X_list[m].T)
            for m in range(self.M)
        ]

        # Σ̂: (batch, M, M),  entry (b,j,k) = (1/T) rⱼ[b] · rₖ[b]
        R = jnp.stack(residuals, axis=1)          # (batch, M, T)
        Sigma_hat = jnp.einsum('bmi,bni->bmn', R, R) / self.N

        _, log_det = jnp.linalg.slogdet(Sigma_hat)
        log_2pi = jnp.log(2.0 * jnp.pi)
        return (-0.5 * self.N * self.M * log_2pi
                - 0.5 * self.N * log_det
                - 0.5 * self.N * self.M)

    def log_prob(self, theta):
        """Profile log-likelihood. theta: (dim,) or (batch, dim) → scalar or (batch,)."""
        theta = jnp.asarray(theta, dtype=jnp.float32)
        scalar_input = theta.ndim == 1
        if scalar_input:
            theta = theta[None, :]
        lp = self._log_lik(theta)
        return lp[0] if scalar_input else lp

    def log_ref(self, x):
        """Log isotropic Gaussian reference density."""
        x = jnp.asarray(x, dtype=jnp.float32)
        ref_var = self.ref_std ** 2
        return (-0.5 * jnp.sum(x ** 2) / ref_var
                - 0.5 * self.dim * jnp.log(2.0 * jnp.pi * ref_var))

    def sample_ref(self, key, n):
        """Draw n i.i.d. samples from the Gaussian reference."""
        import jax.random as jr
        return self.ref_std * jr.normal(key, (n, self.dim), dtype=jnp.float32)

    def score(self, x):
        """Gradient of log_prob at x."""
        return grad(self.log_prob)(x)

    def theta_to_params(self, theta):
        """Identity mapping.

        Uniform 3-tuple interface shared across target classes; this target
        has no separate curve/predictive output.
        """
        return theta, None, None


def make_sur_target(T: int, M: int, k_per_eq: int = 1,
                    seed: int = 0, ref_std: float = 1.0) -> SURTarget:
    """Simulate a SUR dataset and return the profile likelihood as a target.

    For each equation m,
    one covariate vector X_m is drawn i.i.d. N(0,1), and responses are
        Y = X diag(β) + E,   E ~ MN(0, Σ ⊗ I_T)
    with Σ and β drawn randomly from the given seed.

    Parameters
    ----------
    T : int
        Number of observations per equation.
    M : int
        Number of equations (= dimension of β when k_per_eq=1).
    k_per_eq : int
        Number of regressors per equation.
    seed : int
        NumPy random seed used for data generation.
    ref_std : float
        Std of the isotropic Gaussian reference q₀.
    """
    rng = np.random.default_rng(seed)

    A = rng.normal(0, 1, (M, M))
    Sigma_true = A @ A.T / M + np.eye(M)
    beta_true = [rng.normal(0, 1, k_per_eq) for _ in range(M)]
    X_list = [rng.normal(0, 1, (T, k_per_eq)) for _ in range(M)]

    E = rng.multivariate_normal(np.zeros(M), Sigma_true, T)
    Y = np.zeros((T, M))
    for m in range(M):
        Y[:, m] = X_list[m] @ beta_true[m] + E[:, m]

    return SURTarget(Y, X_list, ref_std=ref_std)
