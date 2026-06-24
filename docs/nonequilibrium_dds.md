# Nonequilibrium DDS: Controlled SDE and Comparison with Vanilla DDS

## Overview

This document describes the controlled SDE formulation for the nonequilibrium DDS
(`dds_nonequilibrium.py`) and compares its loss function against the vanilla
equilibrium DDS formulation (not included in this release; the equilibrium
variant is superseded by the nonequilibrium one throughout this codebase, but
the comparison below motivates why). The key idea: replace the stationary OU reference
with the time-reversal of an OU process started from a fitted GMM, so that
the base process already transports N(0, sigma^2 I) close to the target.
The learned control only needs to correct the residual GMM-to-pi mismatch.

---

## 1. Controlled SDE Formulation (Nonequilibrium)

### 1.1 The OU Forward Process

Start from a Gaussian mixture model (GMM) that approximates the target pi:

```
Y_0 ~ p_GMM = sum_k w_k * N(mu_k, Sigma_k)
```

Run an OU process forward:

```
dY_s = -1/2 * Y_s ds + sigma dB_s,   s in [0, T]
```

At time s, the marginal is a GMM with parameters:

```
means_s = e^{-s/2} * mu_k
covs_s  = e^{-s} * Sigma_k + (1 - e^{-s}) * sigma^2 I
```

For large T, the marginal at time T is approximately N(0, sigma^2 I).

### 1.2 The Reverse-OU Base Process

Time-reverse the OU process. Define t = T - s so that t runs from 0 to T.
The reverse process satisfies:

```
dX_t = b_rev(X_t, t) dt + sigma dW_t
```

where the reverse-OU drift is (Anderson, 1982):

```
b_rev(x, t) = 1/2 * x + sigma^2 * grad log p_{T-t}(x)
```

Here p_{T-t} is the OU marginal at forward time s = T - t, which is a GMM
with known parameters. This process maps N(0, sigma^2 I) at t=0
to (approximately) p_GMM at t=T.

### 1.3 Adding the Learned Control

The controlled process adds a correction f_theta to the base drift:

```
dX_t = [ b_rev(X_t, t) + sigma^2 * f_theta(X_t, t) ] dt + sigma dW_t
```

```
     = [ 1/2 * X_t + sigma^2 * grad log p_{T-t}(X_t) + sigma^2 * f_theta(X_t, t) ] dt + sigma dW_t
```

Initial condition: X_0 ~ N(0, sigma^2 I).

When f_theta = 0, the terminal distribution is the GMM.
The role of f_theta is to steer the terminal distribution from the GMM to the
exact target pi.

### 1.4 Discrete-Time Update (Euler-Maruyama)

```
X_{k+1} = X_k + [ b_rev(X_k, t_k) + sigma^2 * f_theta(X_k, t_k) ] * dt_k
         + sigma * sqrt(dt_k) * eps_k
```

where eps_k ~ N(0, I) and dt_k = t_{k+1} - t_k.

### 1.5 Score Parametrization

f_theta uses the PIS-GRAD ansatz (target-augmented parametrization):

```
f_theta(x, t) = NN_state(x, t) + NN_gate(t) * grad log pi(x)
```

This gives the network direct access to the target score, which helps at
times t near T where the process should track pi.

---

## 2. Loss Function (Path-Space KL)

The loss is the path-space KL divergence from the controlled process Q_theta
to the Doob h-transform of the reverse-OU base process Q_0^rev conditioned
on terminal marginal pi.

By Girsanov's theorem, the KL between Q_theta and Q_0^rev decomposes as:

```
L_neq(f_theta) = E_{Q_theta}[ integral_0^T 1/2 * sigma^2 ||f_theta(X_t, t)||^2 dt
                              + log p_GMM(X_T) - log pi(X_T) ]
```

The two terms have clear roles:

- **Running cost** (Girsanov penalty): penalizes the size of the control.
  This is the KL cost of deviating from the reverse-OU base dynamics.
- **Terminal cost**: penalizes the mismatch between the GMM (base terminal
  distribution) and the target pi at the final time.

At optimality (L_neq = 0), the controlled process produces exact samples
from pi at time T.

---

## 3. Comparison with Vanilla (Equilibrium) DDS

### 3.1 Vanilla DDS Setup

The equilibrium DDS uses the OU process itself as the base:

```
Base drift:  b_OU(x) = -1/2 * x
Reference:   p_ref = N(0, sigma^2 I)  (stationary distribution of OU)
Control:     g_theta(x, t)
```

Controlled SDE:

```
dX_t = [ -1/2 * X_t + sigma^2 * g_theta(X_t, t) ] dt + sigma dW_t
```

Loss:

```
L_eq(g_theta) = E_{Q_theta}[ integral_0^T 1/2 * sigma^2 ||g_theta||^2 dt
                             + log p_ref(X_T) - log pi(X_T) ]
```

### 3.2 Same SDE Under Reparametrization

Define the drift offset:

```
h(x, t) = [ b_rev(x, t) - b_OU(x) ] / sigma^2
         = x / sigma^2 + grad log p_{T-t}(x)
```

Setting g_theta = h + f_theta in the equilibrium DDS gives:

```
-1/2 x + sigma^2 (h + f_theta)
= -1/2 x + x + sigma^2 grad log p_{T-t}(x) + sigma^2 f_theta
= b_rev(x, t) + sigma^2 f_theta
```

This is exactly the nonequilibrium controlled SDE. So for any f_theta,
both formulations produce **the same trajectories**.

### 3.3 The Loss Functions Are Not Equivalent

Despite generating identical trajectories, the two losses differ:

```
L_eq - L_neq = E_{Q_theta}[ integral_0^T sigma^2 (1/2 ||h||^2 + h . f_theta) dt
                            + log p_ref(X_T) - log p_GMM(X_T) ]
```

The cross-term h . f_theta makes this **depend on f_theta**, so the two
losses are not related by a constant. They have different gradients
and different optimization landscapes.

### 3.4 Both Are Valid KL Objectives

Each loss equals a KL divergence to a target path measure (the Doob
h-transform of the respective base process conditioned on terminal pi):

```
L_eq   = KL(Q_theta || Q*_OU)  >= 0
L_neq  = KL(Q_theta || Q*_rev) >= 0
```

Both are zero if and only if the terminal marginal of Q_theta equals pi.
The path-level optima Q*_OU and Q*_rev are different processes (different
drifts), but both map N(0, sigma^2 I) to pi at time T.

### 3.5 Summary Table

| | Vanilla DDS (equilibrium) | Nonequilibrium DDS |
|---|---|---|
| Base drift | -1/2 x | 1/2 x + sigma^2 grad log p_{T-t}(x) |
| Base terminal | N(0, sigma^2 I) | GMM approx pi |
| Control | g_theta (full transport) | f_theta (residual correction) |
| Running cost | 1/2 sigma^2 \|\|g_theta\|\|^2 | 1/2 sigma^2 \|\|f_theta\|\|^2 |
| Terminal cost | log p_ref(x_T) - log pi(x_T) | log p_GMM(x_T) - log pi(x_T) |
| Requires GMM | No | Yes |
| OU marginals | Not used | Used in base drift |

---

## 4. Why Nonequilibrium Is Better in Practice

The nonequilibrium formulation has the same global optimum but a
better-conditioned optimization landscape:

1. **Smaller running cost.** Near the optimum, ||f_theta|| is much smaller
   than ||g_theta|| = ||h + f_theta||, because the reverse-OU drift already
   does most of the transport. The equilibrium running cost is dominated by
   the large ||h||^2 term, which adds noise to gradient estimates without
   useful signal.

2. **Smaller terminal gap.** log p_GMM(x_T) - log pi(x_T) is small when the
   GMM is a good approximation, whereas log p_ref(x_T) - log pi(x_T) is
   large (isotropic Gaussian vs. multimodal target).

3. **Lower gradient variance.** The direct gradient of the running cost is
   sigma^2 f_theta (nonequilibrium) vs. sigma^2(h + f_theta) (equilibrium).
   The latter has a large h component that contributes variance without
   improving the optimization direction.

4. **Better initialization.** At f_theta = 0 the nonequilibrium process
   already produces reasonable samples (from the GMM). The equilibrium
   process at g_theta = 0 produces samples from the stationary Gaussian,
   which may be far from pi.


---

## 7. Reference Files

| File | Role |
|------|------|
| `src/sampler/vi/dds_nonequilibrium.py` | Nonequilibrium DDS implementation |
| `src/sampler/vi/utils.py` | Shared tree utilities and training loop |
| `src/sampler/mcmc/mad_path.py` | Metropolis-adjusted proposals using the learned sampler |
