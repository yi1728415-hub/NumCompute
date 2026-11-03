"""Shared input checks so every module reports shape/type problems the same way."""

import numpy as np


def check_2d_numeric(X, name="X", allow_empty=True, allow_nan=True):
    """Return ``X`` as a float64 array of shape (n_samples, n_features).

    A 1-D input raises with a hint, because guessing whether it is one sample
    or one feature silently corrupts results.
    """
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric (convertible to float); "
                        "encode text columns first, e.g. with OneHotEncoder") from exc
    if arr.ndim == 1:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got 1-D shape "
                         f"{arr.shape}; use .reshape(-1, 1) for one feature or "
                         ".reshape(1, -1) for one sample")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}")
    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf; only finite values or NaN are allowed")
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; impute missing values first")
    return arr


def check_n_features(arr, expected, name="X"):
    """Raise if ``arr`` does not have ``expected`` columns (no-op when expected is None)."""
    if expected is not None and arr.shape[1] != expected:
        raise ValueError(f"{name} has {arr.shape[1]} features, but {expected} were seen "
                         "in earlier calls")


def check_1d(y, name="y", n_samples=None):
    """Return ``y`` as a 1-D array; a (n, 1) column is flattened."""
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,), got shape {arr.shape}")
    if n_samples is not None and arr.shape[0] != n_samples:
        raise ValueError(f"{name} has {arr.shape[0]} samples but {n_samples} were expected")
    return arr
