"""Draw-level helpers and the Monte Carlo null model (a fair k-from-N draw)."""
import numpy as np


def indicator(balls, pool):
    """(n_draws, k) array of 1-based balls -> (n_draws, pool) 0/1 matrix."""
    n = balls.shape[0]
    X = np.zeros((n, pool), dtype=np.int8)
    X[np.repeat(np.arange(n), balls.shape[1]), balls.ravel() - 1] = 1
    return X


def simulate_indicator(n_draws, pool, k, rng):
    """One fair history: each draw is k distinct balls from 1..pool."""
    picks = np.argpartition(rng.random((n_draws, pool)), k, axis=1)[:, :k]
    X = np.zeros((n_draws, pool), dtype=np.int8)
    X[np.repeat(np.arange(n_draws), k), picks.ravel()] = 1
    return X


def mc_pvalue(observed, null, greater=True):
    """One-sided Monte Carlo p-value with the +1 correction."""
    null = np.asarray(null)
    hits = (null >= observed).sum() if greater else (null <= observed).sum()
    return (1 + hits) / (1 + len(null))
