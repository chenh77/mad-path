from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

R_OBS     = 0.3    # observation radius
SIGMA_OBS = 0.02   # distance measurement noise std
N_UNKNOWN = 8
N_ANCHORS = 3
N_SENSORS = N_UNKNOWN + N_ANCHORS   # 11

# ---------------------------------------------------------------------------
# Hard-coded dataset read approximately from Ihler et al. (2005) Fig. 1
# ---------------------------------------------------------------------------

IHLER_TRUE_POSITIONS = np.array([
    [0.20, 0.83],   # S0
    [0.56, 0.91],   # S1
    [0.54, 0.75],   # S2
    [0.28, 0.49],   # S3
    [0.80, 0.79],   # S4
    [0.38, 0.12],   # S5
    [0.46, 0.24],   # S6
    [0.80, 0.04],   # S7
], dtype=np.float64)

IHLER_ANCHORS = np.array([
    [0.32, 0.70],   # A0 (index 8)
    [0.50, 0.32],   # A1 (index 9)
    [0.68, 0.70],   # A2 (index 10)
], dtype=np.float64)

IHLER_OBS_PAIRS = np.array([
    [0, 3], [0, 8],
    [1, 2], [1, 8],
    [2, 3], [2, 4], [2, 8], [2, 10], [2, 9],
    [3, 5], [3, 8],
    [4, 6], [4, 10],
    [5, 6], [5, 7],
    [6, 7], [6, 8],
], dtype=np.int32)


class SensorNetworkTarget:
    """Posterior over 8 unknown sensor locations in [0,1]^2.

    Parameters
    ----------
    anchors : array, shape (3, 2)
        Known anchor positions in [0,1]^2.
    obs_pairs : array, shape (M, 2), int
        0-based indices (i, j) with i < j for which a distance was observed.
    obs_distances : array, shape (M,)
        Observed noisy distances y_ij.
    unobs_pairs : array, shape (U, 2), int
        0-based indices (i, j) with i < j for which no distance was observed.
    ref_std : float
        Std of the isotropic Gaussian DDS reference (centred at 0).
    r_obs : float
        Observation radius r (default 0.3).
    sigma_obs : float
        Distance measurement noise std (default 0.02).
    """

    _PARAM_NAMES = [
        s for k in range(N_UNKNOWN) for s in (f'x{k+1}₁', f'x{k+1}₂')
    ]

    def __init__(
        self,
        anchors,
        obs_pairs,
        obs_distances,
        unobs_pairs,
        ref_std: float = 0.3,
        r_obs: float = R_OBS,
        sigma_obs: float = SIGMA_OBS,
    ):
        self.anchors     = jnp.asarray(anchors,       dtype=jnp.float64)  # (3, 2)
        self.obs_pairs   = jnp.asarray(obs_pairs,     dtype=jnp.int32)    # (M, 2)
        self.obs_dists   = jnp.asarray(obs_distances, dtype=jnp.float64)  # (M,)
        self.unobs_pairs = jnp.asarray(unobs_pairs,   dtype=jnp.int32)    # (U, 2)
        self.r_obs     = float(r_obs)
        self.sigma_obs = float(sigma_obs)
        self.ref_std   = float(ref_std)
        self.dim = 2 * N_UNKNOWN   # 16
        self._r2   = 2.0 * r_obs ** 2
        self._sig2 = 2.0 * sigma_obs ** 2

    def _positions(self, theta: jax.Array) -> jax.Array:
        """theta (16,) -> all-sensor positions (11, 2) in [0,1]^2."""
        unknown = theta.reshape(N_UNKNOWN, 2) + 0.5
        return jnp.concatenate([unknown, self.anchors], axis=0)

    def log_prob(self, theta: jax.Array) -> jax.Array:
        """Log unnormalized posterior (improper uniform prior).

        Parameters
        ----------
        theta : (16,) centred coordinates  theta_k = x_k - 0.5

        Returns
        -------
        scalar
        """
        theta = jnp.asarray(theta)
        pos = self._positions(theta)   # (11, 2)

        # ── Observed pairs ──────────────────────────────────────────────
        pi  = pos[self.obs_pairs[:, 0]]            # (M, 2)
        pj  = pos[self.obs_pairs[:, 1]]            # (M, 2)
        d2  = jnp.sum((pi - pj) ** 2, axis=1)     # (M,)
        d   = jnp.sqrt(d2 + 1e-12)
        log_obs = jnp.sum(
            -d2 / self._r2
            - (self.obs_dists - d) ** 2 / self._sig2
        )

        # ── Unobserved pairs: log(1 - exp(-d^2/r^2)) ────────────────────
        # log(1 - exp(-x)) = log(-expm1(-x)), numerically stable for x > 0
        ui  = pos[self.unobs_pairs[:, 0]]          # (U, 2)
        uj  = pos[self.unobs_pairs[:, 1]]          # (U, 2)
        d2u = jnp.sum((ui - uj) ** 2, axis=1)     # (U,)
        log_unobs = jnp.sum(jnp.log(-jnp.expm1(-d2u / self._r2) + 1e-30))

        return log_obs + log_unobs

    def theta_to_params(self, theta: jax.Array):
        """Return (theta, dummy, sigma).

        Uniform 3-tuple interface shared across target classes for
        diagnostic/plotting code; this target has no separate curve output.
        """
        return theta, jnp.zeros(1), jnp.array(self.sigma_obs)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def simulate_data(
    seed: int = 0,
    r_obs: float = R_OBS,
    sigma_obs: float = SIGMA_OBS,
    ref_std: float = 0.1,
    anchors: np.ndarray | None = None,
    true_positions: np.ndarray | None = None,
    obs_pairs: np.ndarray | None = None,
) -> tuple[SensorNetworkTarget, np.ndarray]:
    """Generate a sensor-network localisation dataset.

    Parameters
    ----------
    seed : int
        Random seed for distance noise (and position/edge sampling if not
        hard-coded).
    r_obs, sigma_obs : float
        Observation radius and noise std.
    ref_std : float
        Std of Gaussian DDS reference (passed to SensorNetworkTarget).
    anchors : (3, 2) array or None
        Known anchor positions. Defaults to (0.1,0.1), (0.5,0.9), (0.9,0.1).
    true_positions : (8, 2) array or None
        True unknown sensor positions. If None, drawn uniformly from [0,1]^2.
    obs_pairs : (M, 2) int array or None
        Hard-coded observed pairs (0-based indices, i < j).  If provided the
        probabilistic edge-selection step is skipped; distances are computed
        from true_positions + Gaussian noise (seed still controls the noise).

    Returns
    -------
    target : SensorNetworkTarget
    true_positions : (8, 2) ndarray
    """
    rng = np.random.default_rng(seed)

    if anchors is None:
        anchors = np.array([[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]], dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    if true_positions is None:
        true_positions = rng.uniform(0.0, 1.0, size=(N_UNKNOWN, 2)).astype(np.float32)
    true_positions = np.asarray(true_positions, dtype=np.float64)

    all_pos = np.concatenate([true_positions, anchors], axis=0)  # (11, 2)

    all_pairs = [
        (i, j)
        for i in range(N_UNKNOWN)
        for j in range(i + 1, N_SENSORS)
    ]

    def _arr2d(lst):
        return (np.array(lst, dtype=np.int32) if lst
                else np.zeros((0, 2), dtype=np.int32))

    if obs_pairs is not None:
        obs_pairs_arr = np.asarray(obs_pairs, dtype=np.int32)
        obs_set = set(map(tuple, obs_pairs_arr.tolist()))
        obs_dists_list = []
        for i, j in obs_pairs_arr:
            d = float(np.linalg.norm(all_pos[i] - all_pos[j]))
            obs_dists_list.append(d + rng.normal(0.0, sigma_obs))
        obs_dists_arr   = np.array(obs_dists_list, dtype=np.float64)
        unobs_pairs_arr = _arr2d([[i, j] for i, j in all_pairs if (i, j) not in obs_set])
    else:
        obs_list, obs_dists_list, unobs_list = [], [], []
        r2 = 2.0 * r_obs ** 2
        for i, j in all_pairs:
            d2    = float(np.sum((all_pos[i] - all_pos[j]) ** 2))
            p_obs = float(np.exp(-d2 / r2))
            if rng.uniform() < p_obs:
                obs_list.append([i, j])
                obs_dists_list.append(float(np.sqrt(d2)) + rng.normal(0.0, sigma_obs))
            else:
                unobs_list.append([i, j])
        obs_pairs_arr   = _arr2d(obs_list)
        obs_dists_arr   = (np.array(obs_dists_list, dtype=np.float64) if obs_dists_list
                           else np.zeros(0, dtype=np.float64))
        unobs_pairs_arr = _arr2d(unobs_list)

    print(f"Sensor network: {len(obs_pairs_arr)} observed, {len(unobs_pairs_arr)} unobserved "
          f"out of {len(all_pairs)} pairs")

    target = SensorNetworkTarget(
        anchors, obs_pairs_arr, obs_dists_arr, unobs_pairs_arr,
        ref_std=ref_std, r_obs=r_obs, sigma_obs=sigma_obs,
    )
    return target, true_positions
