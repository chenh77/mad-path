"""Target distribution library for diffusion samplers.

Each module provides a callable target with a unified API:
    target.log_prob(x)       -> scalar log unnormalized density (unconstrained space)
    target.log_ref(x)        -> scalar log reference (Gaussian) density
    target.sample_ref(key,n) -> (n, dim) reference samples
    target.dim               -> int, dimension of the sample space
    target.ref_std           -> float, std of the isotropic Gaussian reference
"""

from target.bayesian_gmm import BayesianGMMTarget
from target.skew import SkewNormalMixtureIID, make_skew_mixture_target
from target.sur import SURTarget, make_sur_target
from target.sensor_network import SensorNetworkTarget, simulate_data as simulate_sensor_network
from target.fetal_deaths import FetalDeathsTarget, make_fetal_deaths_target

__all__ = ["BayesianGMMTarget", "SkewNormalMixtureIID", "make_skew_mixture_target",
           "SURTarget", "make_sur_target",
           "SensorNetworkTarget", "simulate_sensor_network",
           "FetalDeathsTarget", "make_fetal_deaths_target"]
