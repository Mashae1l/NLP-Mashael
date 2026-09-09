"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def _validate(values, n_boot, alpha):
    array = np.asarray(values, dtype=float)

    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    return array


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return the mean and percentile bootstrap confidence interval."""
    array = _validate(values, n_boot, alpha)
    rng = np.random.default_rng(seed)

    indices = rng.integers(
        low=0,
        high=array.size,
        size=(n_boot, array.size),
    )

    bootstrap_means = array[indices].mean(axis=1)
    point = float(array.mean())

    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2, 1 - alpha / 2],
    )

    return point, float(lower), float(upper)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return the paired mean difference and its bootstrap interval."""
    array_a = np.asarray(a, dtype=float)
    array_b = np.asarray(b, dtype=float)

    if array_a.shape != array_b.shape:
        raise ValueError("paired inputs must have matching lengths")

    differences = array_a - array_b

    return bootstrap_ci(
        differences,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )