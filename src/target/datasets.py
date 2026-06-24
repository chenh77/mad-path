"""Dataset loaders for target distribution demos."""

import numpy as np


def load_old_faithful() -> np.ndarray:
    """Load the Old Faithful geyser dataset.

    Returns 272 observations with two columns:
        0 – eruption duration (minutes)
        1 – waiting time to next eruption (minutes)

    Tries ``statsmodels`` first (which bundles R datasets); falls back to a
    synthetic approximation with matching bimodal-Gaussian statistics when
    that package is unavailable.

    Returns
    -------
    Y : np.ndarray, shape (272, 2)
    """
    try:
        import statsmodels.api as sm
        df = sm.datasets.get_rdataset("faithful").data
        return df[["eruptions", "waiting"]].values.astype(np.float64)
    except Exception:
        pass

    # Synthetic fallback: bimodal Gaussian with statistics matching the
    # real faithful dataset (Azzalini & Bowman 1990).
    rng = np.random.default_rng(2024)
    n1, n2 = 97, 175
    cov1 = np.array([[0.070, 0.90], [0.90, 33.8]])
    cov2 = np.array([[0.170, 1.50], [1.50, 36.0]])
    Y1 = rng.multivariate_normal([2.0, 54.0], cov1, n1)
    Y2 = rng.multivariate_normal([4.3, 79.9], cov2, n2)
    Y = np.concatenate([Y1, Y2], axis=0)
    return Y[rng.permutation(n1 + n2)]
