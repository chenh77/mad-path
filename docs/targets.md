# Target distributions

Six target distributions are used in the paper's numerical experiments (Section 6). Each has a
corresponding module in `src/target/` and driver script in `experiments/`.

| Paper section | Target module | Run script |
|---|---|---|
| 6.1 Mixture of Gaussians with unequal variances | defined inline in `experiments/run_mog.py` | `experiments/run_mog.py` |
| 6.2 Mixture of skew normal | `target/skew.py` | `experiments/run_skew.py` |
| 6.3 Bayesian Gaussian mixture model | `target/bayesian_gmm.py` | `experiments/run_bayesian_gmm.py` |
| 6.4 Bayesian mixture of beta-binomial and binomial | `target/fetal_deaths.py` | `experiments/run_fetal_deaths.py` |
| 6.5 Sensor network localization | `target/sensor_network.py` | `experiments/run_sensor_network.py` |
| 6.6 Seemingly unrelated regression | `target/sur.py` | `experiments/run_sur.py` |

## Mixture of Gaussians (`experiments/run_mog.py`)

Isotropic Gaussian mixture target, defined inline in the run script rather than as a reusable
`src/target/` module (its two components and their weights are known exactly, so there is no
separate parameter-fitting step). All observations share a scalar variance per component
(isotropic covariance).

## Mixture of skew normal (`skew.py`)

`SkewNormalMixtureIID`: each mixture component is a skew-normal distribution with
component-specific skewness, applied independently across coordinates.

## Bayesian Gaussian mixture model (`bayesian_gmm.py`)

Posterior for a `K`-component Gaussian mixture model:

```
y_i | z_i, mu, Sigma  ~  N_p(mu_{z_i}, Sigma_{z_i})
mu_k                  ~  N_p(m_0, V_0)
Sigma_k^{-1}          ~  Wishart(nu_0, W_0)
pi                    ~  Dirichlet(alpha_0 * 1_K)
```

The posterior is evaluated in a fully unconstrained space of dimension
`K*p + K*p*(p+1)//2 + K-1`:

- `theta[0 : K*p]` — means `mu_k` (flattened, unconstrained).
- `theta[K*p : K*p + K*pL]` — log-Cholesky factor of each precision matrix
  (`pL = p*(p+1)//2` per component); diagonal entries are stored as `log(L[j,j])`, off-diagonals
  stored directly.
- `theta[K*p + K*pL :]` — `K-1` free logits for `pi`; the `K`-th logit is anchored at 0 and
  `pi = softmax(logits)`.

The log-density includes the log-Jacobian of this change of variables so that sampling
`theta ~ p_theta` is equivalent to sampling `(mu, Prec, pi) ~ posterior`:

- Log-Cholesky parameterization (`l -> L -> Prec = L L^T`): combining the Jacobian
  `log|J_k| = sum_j (p-j+1) * l_k_diag[j] + p*log2` with the Wishart density gives a net
  coefficient of `(nu_0 - j)` on `log(L_k[j,j])`, i.e. component-`k` prior contribution
  `sum_j (nu_0 - j) * l_k_diag[j] - 1/2 * tr(W_0^{-1} L_k L_k^T)`.
- Dirichlet–softmax parameterization: `log|J_pi| = sum_k log(pi_k)`, giving weight-prior
  contribution `alpha_0 * sum_k log(pi_k)`.

Applied to the Old Faithful geyser dataset in the paper.

## Fetal deaths in litters (`fetal_deaths.py`)

Bimodal beta-binomial/binomial mixture posterior from Tjelmeland & Hegstad (2001), Section 4.2,
reproducing the analysis of Brooks et al. (1997) on the HS3 dataset.

```
p(x | n, gamma, mu, theta, nu) = gamma * p_bb(x | n, mu, theta) + (1-gamma) * p_bin(x | n, nu)
```

where, for a litter of size `n` with `x` observed deaths,

```
p_bb(x | n, mu, theta) = C(n,x) * Gamma(x+mu/theta)/Gamma(mu/theta)
                          * Gamma(n-x+(1-mu)/theta)/Gamma((1-mu)/theta)
                          / [Gamma(n+1/theta)/Gamma(1/theta)]      (beta-binomial)
p_bin(x | n, nu)       = C(n,x) * nu^x * (1-nu)^(n-x)              (binomial)
```

with mixing weight `gamma in (0,1)`, beta-binomial mean `mu in (0,1)`, overdispersion
`theta >= 0` (`theta -> 0` recovers binomial with mean `mu`), and binomial success probability
`nu in (0,1)`.

Priors: `gamma, mu, nu ~ Uniform(0,1)`, `theta ~ Exponential(1)`. Sampling is done in the
unconstrained space `(gamma~, mu~, theta~, nu~) = (logit(gamma), logit(mu), log(theta), logit(nu))`,
with corresponding log-Jacobian terms `log(gamma(1-gamma))`, `log(mu(1-mu))`, `theta~`,
`log(nu(1-nu))`.

Data: HS3 dataset from Brooks et al. (1997), Table 1 in Tjelmeland & Hegstad (2001), with columns
litter size `n`, dead count `x`, and number of litters. The posterior has two known modes
(~97.9% and ~2.1% of the probability mass).

## Sensor network localization (`sensor_network.py`)

11 sensors `x_1, ..., x_11` scattered on `[0,1]^2`: `x_1, ..., x_8` are unknown locations to infer,
`x_9, x_10, x_11` are known anchors. For every pair `(i,j)` with `i` unknown, a distance is
*observed* with probability `p_obs = exp(-||x_i - x_j||^2 / (2 r^2))`, `r = 0.25`; when observed,
`y_ij ~ N(||x_i - x_j||, sigma_obs^2)` with `sigma_obs = 0.01`, contributing log-factor
`log p_obs + log N(y_ij; d_ij, sigma_obs^2)`; when unobserved, the log-factor is
`log(1 - p_obs)`. Both unknown-unknown and unknown-anchor pairs are included
(`C(8,2)=28 + 8*3=24 = 52` pairs total).

The log-posterior under an improper uniform prior on each coordinate is
`sum_{(i,j)} log f_ij(x_i, x_j)`. Coordinates are parameterized as `theta_k = x_k - 0.5` so that
`theta ~ 0` is the center of `[0,1]^2`, matching a zero-centered Gaussian DDS reference; total
dimension `D = 16`.

Problem originally from Ihler et al. (2005, JMLR); analyzed in several MCMC-for-multimodal papers.

## Seemingly unrelated regression (`sur.py`)

`SURTarget`: profile-likelihood target for the SUR model. Each response `m` is regressed on its
own covariate set, `y_m = X_m beta_m + eps_m`, with `eps ~ MN(0, Sigma kron I_N)`. The profile
likelihood integrates out `Sigma` analytically:

```
l(beta) = -(N*M/2) log(2*pi) - (N/2) log|Sigma_hat(beta)| - N*M/2
```

where `Sigma_hat(beta)_{jk} = (1/N) r_j(beta)^T r_k(beta)` and `r_m(beta) = y_m - X_m beta_m`.
