"""Shared utilities for variational inference samplers."""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax


# ============================================================
# Tree operations
# ============================================================
def tree_weighted_mean(w, tree):
    """Weighted mean over the leading axis of each leaf, with weights *w*."""
    def _fn(leaf):
        return jnp.mean(w.reshape((-1,) + (1,) * (leaf.ndim - 1)) * leaf, axis=0)
    return jax.tree.map(_fn, tree)


def tree_mean(tree):
    """Unweighted mean over the leading axis of each leaf."""
    return jax.tree.map(lambda x: jnp.mean(x, axis=0), tree)


def tree_add(a, b):
    return jax.tree.map(jnp.add, a, b)


def tree_scale(c, t):
    return jax.tree.map(lambda x: c * x, t)


def flatten_tree(tree):
    return jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree.leaves(tree)])


def tree_num_params(tree):
    return sum(leaf.size for leaf in jax.tree.leaves(tree))


# ============================================================
# Importance sampling
# ============================================================
def compute_ess(log_weights):
    """Effective sample size from unnormalized log-weights."""
    lw = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jnp.exp(-jax.scipy.special.logsumexp(2 * lw))


# ============================================================
# Laplacian probe for Hutchinson estimator
# ============================================================
def sample_laplacian_probe(key, shape):
    return 2.0 * jr.bernoulli(key, 0.5, shape=shape).astype(jnp.float32) - 1.0


# ============================================================
# Training loop
# ============================================================
def train(params0, loss_and_grad_fn, n_iter, *,
          lr=3e-3, grad_clip=10.0, log_every=10, key=None,
          extra_metric_fn=None):
    """Generic training loop.

    *loss_and_grad_fn(params, key)* must return ``(loss, aux, grads)``
    where *aux* is a scalar diagnostic (e.g. observable variance).

    Returns ``(params, metrics)`` where *metrics* is a dict with keys
    ``iters``, ``losses``, ``aux_vals``, ``diag_vals``, ``wall``.
    """
    params = jax.tree.map(lambda x: x.copy(), params0)
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = opt.init(params)
    iters, losses, aux_vals = [], [], []
    diag_vals = {}
    if key is None:
        key = jr.PRNGKey(123)
    t0 = time.time()
    for i in range(n_iter):
        key, sk = jr.split(key)
        loss, aux, grads = loss_and_grad_fn(params, sk)
        updates, opt_state = opt.update(grads, opt_state, params)
        if i % log_every == 0:
            iters.append(i)
            losses.append(float(loss))
            aux_vals.append(float(aux))
            if extra_metric_fn is not None:
                sk, mk = jr.split(sk)
                extras = extra_metric_fn(params, mk)
                for name, value in extras.items():
                    diag_vals.setdefault(name, []).append(float(value))
        params = optax.apply_updates(params, updates)
    wall = time.time() - t0
    return params, {
        "iters": iters,
        "losses": losses,
        "aux_vals": aux_vals,
        "diag_vals": diag_vals,
        "wall": wall,
    }
