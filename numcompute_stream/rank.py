"""Ranking with ties and percentiles.

Functions
---------
rank        Rank data along an axis ('average', 'dense', 'ordinal', 'min', 'max').
percentile  q-th percentile with several interpolation rules and NaN policies.

The private helper ``_quantile`` is shared with ``stats.quantile`` so that
percentiles and quantiles always agree.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rank", "percentile"]

_RANK_METHODS = ("average", "dense", "ordinal", "min", "max")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")
_NAN_POLICIES = ("propagate", "omit", "raise")


def rank(data, method="average", axis=-1, nan_policy="keep"):
    """Assign ranks to data, handling ties. Ranks start at 1.

    Example for ``[10, 20, 20, 30]``:

    ========= ==================
    method    ranks
    ========= ==================
    average   [1, 2.5, 2.5, 4]
    min       [1, 2, 2, 4]
    max       [1, 3, 3, 4]
    dense     [1, 2, 2, 3]
    ordinal   [1, 2, 3, 4]   (ties broken by original position)
    ========= ==================

    Parameters
    ----------
    data : array_like, shape (..., n, ...)
    method : {'average', 'dense', 'ordinal', 'min', 'max'}, default 'average'
    axis : int or None, default -1
        Axis along which to rank; each 1-D slice is ranked independently.
        ``None`` ranks the flattened array (result is 1-D).
    nan_policy : {'keep', 'raise'}, default 'keep'
        'keep' gives NaN a rank of NaN and ranks the other values as if the
        NaNs were absent. 'raise' raises ValueError if any NaN is present.

    Returns
    -------
    ranks : ndarray, same shape as ``data``
        float64 for 'average' or when NaNs are kept, otherwise int64.

    Raises
    ------
    ValueError If ``method``/``nan_policy`` is invalid, ``data`` is a scalar,
               or NaN is found with ``nan_policy='raise'``.
    AxisError  If ``axis`` is out of range.

    Notes
    -----
    Fully vectorised: one stable argsort, then group boundaries are found
    with ``diff`` and propagated with ``maximum.accumulate`` /
    ``minimum.accumulate``. Time O(n log n), space O(n).
    """
    if method not in _RANK_METHODS:
        raise ValueError(f"method must be one of {_RANK_METHODS}, got {method!r}")
    if nan_policy not in ("keep", "raise"):
        raise ValueError(f"nan_policy must be 'keep' or 'raise', got {nan_policy!r}")
    a = np.asarray(data)
    if a.dtype == object:
        raise TypeError("data must have a numeric or string dtype, got object")
    if a.ndim == 0:
        raise ValueError("cannot rank a scalar; pass a 1-D or N-D array")
    if axis is None:
        a, axis = a.ravel(), 0
    if not -a.ndim <= axis < a.ndim:
        raise np.exceptions.AxisError(axis, a.ndim)

    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    out_dtype = np.float64 if method == "average" else np.int64
    if n == 0:
        return np.moveaxis(np.empty(a.shape, dtype=out_dtype), -1, axis)

    if np.issubdtype(a.dtype, np.floating):
        nan_mask = np.isnan(a)
    else:
        nan_mask = np.zeros(a.shape, dtype=bool)
    if nan_policy == "raise" and nan_mask.any():
        raise ValueError("data contains NaN (nan_policy='raise')")

    # 1) stable sort; NaNs go to the end so they do not disturb other ranks
    sorter = np.argsort(a, axis=-1, kind="stable")
    s = np.take_along_axis(a, sorter, axis=-1)

    # 2) mark the first element of every group of equal values
    starts = np.ones(s.shape, dtype=bool)
    starts[..., 1:] = s[..., 1:] != s[..., :-1]
    pos = np.broadcast_to(np.arange(n), s.shape)

    if method == "ordinal":
        r_sorted = pos + 1
    elif method == "dense":
        r_sorted = np.cumsum(starts, axis=-1)
    else:
        # position of the first element of my group
        first = np.maximum.accumulate(np.where(starts, pos, 0), axis=-1)
        # position of the last element of my group
        ends = np.ones(s.shape, dtype=bool)
        ends[..., :-1] = starts[..., 1:]
        last = np.flip(
            np.minimum.accumulate(np.flip(np.where(ends, pos, n - 1), -1), axis=-1),
            -1,
        )
        if method == "min":
            r_sorted = first + 1
        elif method == "max":
            r_sorted = last + 1
        else:  # average
            r_sorted = (first + last) / 2.0 + 1.0

    # 3) scatter ranks back to the original positions
    ranks = np.empty(s.shape, dtype=out_dtype)
    np.put_along_axis(ranks, sorter, r_sorted.astype(out_dtype), axis=-1)

    if nan_mask.any():
        ranks = ranks.astype(np.float64)
        ranks[nan_mask] = np.nan
    return np.moveaxis(ranks, -1, axis)


def _quantile(data, q_frac, interpolation="linear", axis=None, nan_policy="propagate"):
    """Shared quantile engine; ``q_frac`` is in [0, 1].

    Returns a float if both ``q_frac`` and the reduced result are scalar,
    otherwise an ndarray of shape ``q.shape + reduced_shape``.
    """
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {_INTERPOLATIONS}, got {interpolation!r}"
        )
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    a = np.asarray(data)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {a.dtype}")
    a = a.astype(np.float64)
    qf = np.asarray(q_frac, dtype=np.float64)
    if np.isnan(qf).any() or (qf < 0).any() or (qf > 1).any():
        raise ValueError("q is out of range")

    if axis is None:
        a, axis = a.ravel(), 0
    if a.ndim == 0:
        raise ValueError("data must have at least 1 dimension")
    if not -a.ndim <= axis < a.ndim:
        raise np.exceptions.AxisError(axis, a.ndim)
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    if n == 0:
        raise ValueError("cannot compute a percentile/quantile of an empty array")

    nan_mask = np.isnan(a)
    has_nan = nan_mask.any(axis=-1)  # shape: batch
    if nan_policy == "raise" and has_nan.any():
        raise ValueError("data contains NaN (nan_policy='raise')")

    s = np.sort(a, axis=-1)  # NaNs sorted to the end
    batch = s.shape[:-1]
    if nan_policy == "omit":
        n_valid = n - nan_mask.sum(axis=-1)
    else:
        n_valid = np.full(batch, n)

    # pos has shape q.shape + batch
    q_b = qf.reshape(qf.shape + (1,) * len(batch))
    pos = q_b * (n_valid - 1)
    pos = np.maximum(pos, 0.0)  # n_valid == 0 -> handled below
    # Snap values like 3.0000000000000004 (from q/100 rounding) to 3.0 so
    # 'higher'/'midpoint' do not jump to the next element by mistake.
    near = np.around(pos)
    pos = np.where(np.abs(pos - near) < 1e-9, near, pos)
    if interpolation == "nearest":
        lo = hi = np.around(pos).astype(np.intp)  # round half to even, like NumPy
    else:
        lo = np.floor(pos).astype(np.intp)
        hi = np.ceil(pos).astype(np.intp)
    frac = pos - lo
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 0, n - 1)

    s_b = np.broadcast_to(s, qf.shape + s.shape)
    v_lo = np.take_along_axis(s_b, lo[..., None], axis=-1)[..., 0]
    v_hi = np.take_along_axis(s_b, hi[..., None], axis=-1)[..., 0]

    with np.errstate(invalid="ignore"):
        if interpolation in ("lower", "nearest"):
            res = v_lo
        elif interpolation == "higher":
            res = v_hi
        elif interpolation == "midpoint":
            res = np.where(lo == hi, v_lo, (v_lo + v_hi) / 2.0)
        else:  # linear
            res = np.where(frac == 0, v_lo, v_lo + (v_hi - v_lo) * frac)

    if nan_policy == "propagate":
        res = np.where(has_nan, np.nan, res)
    else:
        res = np.where(n_valid == 0, np.nan, res)

    if np.ndim(res) == 0:
        return float(res)
    return np.asarray(res, dtype=np.float64)


def percentile(data, q, interpolation="linear", axis=None, nan_policy="propagate"):
    """Compute the q-th percentile of the data.

    For sorted data ``s`` of length ``n``, the target position is
    ``p = q/100 * (n - 1)``. With ``i = floor(p)``, ``j = ceil(p)``:

    - 'linear'   : ``s[i] + (s[j] - s[i]) * (p - i)``
    - 'lower'    : ``s[i]``
    - 'higher'   : ``s[j]``
    - 'midpoint' : ``(s[i] + s[j]) / 2``
    - 'nearest'  : ``s[round(p)]`` (half to even)

    These match ``np.percentile(..., method=...)``.

    Parameters
    ----------
    data : array_like, shape (..., n, ...)
    q : float or array_like of float, each in [0, 100]
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    axis : int or None, default None
        ``None`` uses the flattened array.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN, 'omit'
        ignores NaNs (an all-NaN slice gives NaN), 'raise' raises.

    Returns
    -------
    float, or ndarray of shape ``q.shape + reduced_shape`` where
    ``reduced_shape`` is ``data.shape`` without ``axis``.

    Raises
    ------
    ValueError If ``q`` is outside [0, 100], ``data`` is empty, an option
               is invalid, or NaN is found with ``nan_policy='raise'``.
    TypeError  If ``data`` is not numeric.

    Notes
    -----
    Time O(n log n) per slice (one sort serves all q), space O(n * len(q)).
    """
    q = np.asarray(q, dtype=np.float64)
    if np.isnan(q).any() or (q < 0).any() or (q > 100).any():
        raise ValueError("q must be in [0, 100]")
    return _quantile(data, q / 100.0, interpolation, axis, nan_policy)
