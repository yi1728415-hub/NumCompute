"""Descriptive statistics, histograms, quantiles and streaming statistics.

Axis convention (same for every function here)
----------------------------------------------
``axis=None`` reduces over all elements and returns a Python ``float``.
``axis=k`` reduces over dimension ``k`` and returns an array whose shape is
``data.shape`` with dimension ``k`` removed. For a data matrix of shape
``(n_samples, n_features)``, ``axis=0`` gives one value per feature.

NaN policy (same for every function that accepts ``nan_policy``)
-----------------------------------------------------------------
'propagate' (default): any NaN in a slice makes that result NaN.
'omit'               : NaNs are ignored; an all-NaN slice gives NaN.
'raise'              : ValueError if any NaN is present.

Functions / classes
-------------------
Batch (whole-array) statistics, kept from Assignment 2.1:
mean, median, var, std, minimum, maximum, describe, quantile, histogram,
RunningStats (Welford / Chan streaming mean and variance).

Streaming statistics, added for Assignment 2.2:
merge_moments   Chan et al. merge of two (count, mean, M2) summaries.
chunk_moments   NaN-aware per-feature (count, mean, M2) of one chunk.
StreamingHistogram  fixed-memory histogram whose range grows by doubling,
                    with an optional sliding window of recent chunks.
StreamingStats  per-feature count / NaN count / mean / var / std / min / max /
                quantiles, updated chunk by chunk with ``update_stats``.
"""

from __future__ import annotations

import warnings

import numpy as np

from .rank import _quantile

__all__ = [
    "mean",
    "median",
    "var",
    "std",
    "minimum",
    "maximum",
    "describe",
    "quantile",
    "histogram",
    "RunningStats",
    "merge_moments",
    "chunk_moments",
    "StreamingHistogram",
    "StreamingStats",
]

_NAN_POLICIES = ("propagate", "omit", "raise")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _prepare(data, axis, nan_policy):
    """Validate input and return a float64 array."""
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    a = np.asarray(data)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {a.dtype}")
    if np.iscomplexobj(a):
        raise TypeError("complex data is not supported")
    a = a.astype(np.float64)
    if axis is not None:
        if a.ndim == 0:
            raise ValueError("axis given but data is a scalar")
        if not -a.ndim <= axis < a.ndim:
            raise np.exceptions.AxisError(axis, a.ndim)
        if a.shape[axis] == 0:
            raise ValueError(f"cannot reduce over empty axis {axis} (shape {a.shape})")
    elif a.size == 0:
        raise ValueError("cannot compute statistics of an empty array")
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("data contains NaN (nan_policy='raise')")
    return a


def _finish(res):
    """Return a Python float for 0-d results, otherwise a float64 ndarray."""
    res = np.asarray(res, dtype=np.float64)
    return float(res) if res.ndim == 0 else res


def _reduce(func, nanfunc, data, axis, nan_policy, **kw):
    a = _prepare(data, axis, nan_policy)
    f = nanfunc if nan_policy == "omit" else func
    # All-NaN slices with 'omit' and ddof >= n produce NaN on purpose;
    # silence NumPy's RuntimeWarnings for those documented cases.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _finish(f(a, axis=axis, **kw))


# --------------------------------------------------------------------------
# descriptive statistics
# --------------------------------------------------------------------------
def mean(data, axis=None, nan_policy="propagate"):
    """Arithmetic mean.

    Parameters
    ----------
    data : array_like (numeric)
    axis : int or None, default None
    nan_policy : {'propagate', 'omit', 'raise'}

    Returns
    -------
    float or ndarray (shape = data.shape without ``axis``)

    Raises
    ------
    ValueError Empty input / empty axis, bad nan_policy, NaN with 'raise'.
    TypeError  Non-numeric input.
    AxisError  Invalid axis.

    Notes
    -----
    Time O(n), space O(1) extra beyond the float64 copy.
    """
    return _reduce(np.mean, np.nanmean, data, axis, nan_policy)


def median(data, axis=None, nan_policy="propagate"):
    """Median (50th percentile, linear interpolation for even n).

    Same parameters, returns and exceptions as :func:`mean`.
    Time O(n) average (NumPy uses partitioning), space O(n).
    """
    return _reduce(np.median, np.nanmedian, data, axis, nan_policy)


def var(data, axis=None, ddof=0, nan_policy="propagate"):
    """Variance ``sum((x - mean)**2) / (n - ddof)``.

    Parameters
    ----------
    ddof : int, default 0
        0 gives the population variance, 1 the sample variance.
        If ``n - ddof <= 0`` the result is NaN.

    Other parameters, returns and exceptions as :func:`mean`.
    Time O(n), space O(n).
    """
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return _reduce(np.var, np.nanvar, data, axis, nan_policy, ddof=ddof)


def std(data, axis=None, ddof=0, nan_policy="propagate"):
    """Standard deviation, ``sqrt(var(data, ddof))``. See :func:`var`."""
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return _reduce(np.std, np.nanstd, data, axis, nan_policy, ddof=ddof)


def minimum(data, axis=None, nan_policy="propagate"):
    """Smallest value. See :func:`mean`. Time O(n)."""
    return _reduce(np.min, np.nanmin, data, axis, nan_policy)


def maximum(data, axis=None, nan_policy="propagate"):
    """Largest value. See :func:`mean`. Time O(n)."""
    return _reduce(np.max, np.nanmax, data, axis, nan_policy)


def quantile(data, q, axis=None, interpolation="linear", nan_policy="propagate"):
    """q-th quantile, with ``q`` in [0, 1].

    ``quantile(x, 0.25)`` equals ``rank.percentile(x, 25)``; both use the
    same engine, so the interpolation rules are identical
    ('linear', 'lower', 'higher', 'midpoint', 'nearest').

    Returns
    -------
    float, or ndarray of shape ``q.shape + reduced_shape``.

    Raises
    ------
    ValueError If any q is outside [0, 1], input is empty, an option is
               invalid, or NaN is found with ``nan_policy='raise'``.

    Notes
    -----
    Time O(n log n) per slice, space O(n * len(q)).
    """
    return _quantile(data, q, interpolation, axis, nan_policy)


def describe(data, axis=0, ddof=1, nan_policy="propagate"):
    """Summary statistics in one call.

    Parameters
    ----------
    data : array_like
        Typically shape (n_samples, n_features) with ``axis=0``.
    axis : int or None, default 0
    ddof : int, default 1 (sample standard deviation)
    nan_policy : {'propagate', 'omit', 'raise'}

    Returns
    -------
    dict with keys 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'.
    'count' is the number of non-NaN values; the others are floats or
    arrays of shape ``data.shape`` without ``axis``.
    """
    a = _prepare(data, axis, nan_policy)
    count = np.sum(~np.isnan(a), axis=axis)
    q = quantile(a, [0.25, 0.5, 0.75], axis=axis, nan_policy=nan_policy)
    return {
        "count": int(count) if np.ndim(count) == 0 else count,
        "mean": mean(a, axis, nan_policy),
        "std": std(a, axis, ddof, nan_policy),
        "min": minimum(a, axis, nan_policy),
        "25%": _finish(q[0]),
        "50%": _finish(q[1]),
        "75%": _finish(q[2]),
        "max": maximum(a, axis, nan_policy),
    }


# --------------------------------------------------------------------------
# histogram
# --------------------------------------------------------------------------
def histogram(data, bins=10, range=None, density=False):
    """Count how many values fall into each bin.

    Bins are half-open ``[e_i, e_{i+1})`` except the last, which also
    includes its right edge (same convention as ``np.histogram``).

    Parameters
    ----------
    data : array_like (any shape, flattened)
        NaNs are always ignored.
    bins : int or array_like of float, default 10
        Number of equal-width bins, or the bin edges (1-D, increasing,
        at least 2 values).
    range : (float, float) or None
        Lower and upper edge when ``bins`` is an int. Default is
        ``(min, max)`` of the data. Values outside are ignored.
    density : bool, default False
        If True, return a probability density (area sums to 1).

    Returns
    -------
    counts : ndarray, shape (n_bins,)
        int64, or float64 if ``density``.
    edges : ndarray of float64, shape (n_bins + 1,)

    Raises
    ------
    ValueError Invalid ``bins``/``range``, or infinite values when the
               range must be inferred.
    TypeError  Non-numeric input.

    Notes
    -----
    If all values are equal the range is widened to ``(v - 0.5, v + 0.5)``.
    Bin indices are computed arithmetically for equal-width bins (O(n)) or
    with ``np.searchsorted`` for custom edges (O(n log b)); counts come
    from ``np.bincount``. Space O(n + b).
    """
    a = np.asarray(data)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {a.dtype}")
    a = a.astype(np.float64).ravel()
    a = a[~np.isnan(a)]

    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        nb = int(bins)
        if nb < 1:
            raise ValueError(f"bins must be >= 1, got {nb}")
        if range is None:
            if a.size and not np.isfinite(a).all():
                raise ValueError("data contains inf; pass an explicit range")
            lo, hi = (a.min(), a.max()) if a.size else (0.0, 1.0)
        else:
            lo, hi = float(range[0]), float(range[1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise ValueError(f"range must be finite with min <= max, got {range}")
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, nb + 1)
        x = a[(a >= lo) & (a <= hi)]
        idx = ((x - lo) / (hi - lo) * nb).astype(np.intp)
        idx = np.clip(idx, 0, nb - 1)
        # correct floating-point rounding at bin boundaries
        idx -= (x < edges[idx]).astype(np.intp)
        idx += ((x >= edges[idx + 1]) & (idx != nb - 1)).astype(np.intp)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin edges must be a 1-D array with at least 2 values")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be strictly increasing")
        nb = edges.size - 1
        x = a[(a >= edges[0]) & (a <= edges[-1])]
        idx = np.searchsorted(edges, x, side="right") - 1
        idx[idx == nb] = nb - 1  # right edge belongs to the last bin

    counts = np.bincount(idx, minlength=nb).astype(np.int64)
    if density:
        total = counts.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return counts / (total * np.diff(edges)), edges
    return counts, edges


# --------------------------------------------------------------------------
# streaming statistics
# --------------------------------------------------------------------------
class RunningStats:
    """Streaming mean / variance / min / max (Welford + Chan's merge).

    Feed data in batches with :meth:`update`; the statistics equal those of
    all data seen so far, without keeping the data in memory. Each batch is
    processed with vectorised NumPy, then merged with the running totals:

        delta = mean_b - mean
        mean  = mean + delta * n_b / (n + n_b)
        M2    = M2 + M2_b + delta**2 * n * n_b / (n + n_b)

    This is numerically stable (no ``sum(x**2) - n*mean**2`` cancellation).

    Parameters
    ----------
    ddof : int, default 0
        Delta degrees of freedom used by :attr:`var` and :attr:`std`.

    Examples
    --------
    >>> rs = RunningStats()
    >>> for chunk in np.array_split(np.arange(10.0), 3):
    ...     rs.update(chunk)
    >>> rs.mean, rs.count
    (4.5, 10)

    Notes
    -----
    A batch of shape (n_b,) is n_b scalar samples. A batch of shape
    (n_b, d1, ...) is n_b samples of shape (d1, ...); statistics then have
    shape (d1, ...). Time O(n_b) per update, space O(d).
    """

    def __init__(self, ddof=0):
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0, got {ddof}")
        self.ddof = ddof
        self._n = 0
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None

    def update(self, batch):
        """Add a batch of samples (first axis = samples). Returns ``self``.

        Raises
        ------
        ValueError If the batch contains NaN or its feature shape differs
                   from earlier batches.
        TypeError  If the batch is not numeric.
        """
        x = np.asarray(batch)
        if not (np.issubdtype(x.dtype, np.number) or x.dtype == bool):
            raise TypeError(f"batch must be numeric, got dtype {x.dtype}")
        x = np.atleast_1d(x.astype(np.float64))
        if np.isnan(x).any():
            raise ValueError("batch contains NaN; impute or drop NaNs first")
        nb = x.shape[0]
        if nb == 0:
            return self
        if self._n and x.shape[1:] != self._mean.shape:
            raise ValueError(
                f"feature shape {x.shape[1:]} does not match earlier {self._mean.shape}"
            )
        b_mean = x.mean(axis=0)
        b_m2 = ((x - b_mean) ** 2).sum(axis=0)
        b_min, b_max = x.min(axis=0), x.max(axis=0)
        if self._n == 0:
            self._mean, self._m2, self._min, self._max = b_mean, b_m2, b_min, b_max
        else:
            tot = self._n + nb
            delta = b_mean - self._mean
            self._mean = self._mean + delta * (nb / tot)
            self._m2 = self._m2 + b_m2 + delta**2 * (self._n * nb / tot)
            self._min = np.minimum(self._min, b_min)
            self._max = np.maximum(self._max, b_max)
        self._n += nb
        return self

    def _check(self):
        if self._n == 0:
            raise ValueError("no data seen yet; call update() first")

    @property
    def count(self):
        """Number of samples seen."""
        return self._n

    @property
    def mean(self):
        """Return a copy of the running mean; raise ValueError before any update."""
        self._check()
        return _finish(self._mean)

    @property
    def var(self):
        """Return variance using the configured ddof; insufficient degrees of freedom yield NaN."""
        self._check()
        denom = self._n - self.ddof
        return _finish(self._m2 / denom if denom > 0 else np.full_like(self._m2, np.nan))

    @property
    def std(self):
        """Return the square root of the running variance."""
        return _finish(np.sqrt(self.var))

    @property
    def min(self):
        """Return a copy of the running minimum; raise ValueError before any update."""
        self._check()
        return _finish(self._min)

    @property
    def max(self):
        """Return a copy of the running maximum; raise ValueError before any update."""
        self._check()
        return _finish(self._max)


# --------------------------------------------------------------------------
# chunk-based (streaming) statistics
# --------------------------------------------------------------------------
from collections import deque  # noqa: E402  (kept next to the code that uses it)


def merge_moments(n_a, mean_a, m2_a, n_b, mean_b, m2_b):
    """Merge two (count, mean, M2) summaries into one (Chan et al., 1979).

    ``M2`` is the sum of squared deviations from the mean, so the population
    variance is ``M2 / n``. Works element-wise on arrays (one entry per
    feature) and handles groups with zero samples::

        delta = mean_b - mean_a
        n     = n_a + n_b
        mean  = mean_a + delta * n_b / n
        M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

    Unlike ``sum(x**2) - n * mean**2`` this never subtracts two large, nearly
    equal numbers, so it stays accurate when the mean is large compared with
    the spread (e.g. timestamps).

    Returns
    -------
    (n, mean, M2) with the broadcast shape of the inputs. Where ``n == 0`` the
    mean and M2 are 0.
    """
    n_a = np.asarray(n_a, dtype=np.float64)
    n_b = np.asarray(n_b, dtype=np.float64)
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    n = n_a + n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_b = np.where(n > 0, n_b / n, 0.0)
        cross = np.where(n > 0, n_a * n_b / n, 0.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * frac_b
    m2 = np.asarray(m2_a, dtype=np.float64) + np.asarray(m2_b, dtype=np.float64) + delta ** 2 * cross
    return n, mean, m2


def chunk_moments(X):
    """Per-feature (count, mean, M2) of one chunk, ignoring NaN.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)

    Returns
    -------
    count, mean, M2 : arrays of shape (n_features,)
        An all-NaN feature gets count 0, mean 0 and M2 0.
    """
    X = np.asarray(X, dtype=np.float64)
    ok = ~np.isnan(X)
    count = ok.sum(axis=0).astype(np.float64)
    total = np.where(ok, X, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    dev = np.where(ok, X - mean, 0.0)
    m2 = (dev * dev).sum(axis=0)
    return count, mean, m2


class StreamingHistogram:
    """Fixed-memory histogram for a 1-D stream of values.

    Parameters
    ----------
    bins : int, default 32
        Number of equal-width bins (rounded up to an even number so that two
        neighbouring bins can always be merged into one).
    range : (float, float) or None
        Fixed lower/upper edge. Values outside are counted in ``n_outside_``
        and left out of the bins. If None, the range is taken from the first
        chunk and **grows by doubling** whenever a later value falls outside:
        the bin width doubles and every pair of neighbouring bins is merged,
        so no count is ever lost (only resolution).
    window : int or None
        If set, only the most recent ``window`` chunks are counted
        (a sliding-window histogram). None = everything seen so far.

    Notes
    -----
    Memory is O(bins * window). ``update`` is O(n_chunk) with ``np.bincount``.
    NaN values are counted in ``n_nan_`` and ignored.
    """

    def __init__(self, bins=32, range=None, window=None):
        if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2:
            raise ValueError(f"bins must be an int >= 2, got {bins!r}")
        if window is not None and (not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError(f"window must be None or an int >= 1, got {window!r}")
        self.bins = int(bins) + int(bins) % 2
        self.window = window
        self.fixed_range = range is not None
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"range must be finite with min < max, got {range}")
            self._lo, self._hi = lo, hi
        else:
            self._lo = self._hi = None
        self._chunks = deque()            # per-chunk counts (window mode)
        self._total = np.zeros(self.bins, dtype=np.int64)
        self.n_nan_ = 0
        self.n_outside_ = 0

    # -- helpers ----------------------------------------------------------
    @property
    def edges(self):
        """Bin edges, shape (bins + 1,). None before the first value."""
        if self._lo is None:
            return None
        return np.linspace(self._lo, self._hi, self.bins + 1)

    def _grow_to(self, lo, hi):
        """Double the range until it covers [lo, hi], merging bin pairs."""
        while lo < self._lo or hi >= self._hi:
            width = self._hi - self._lo
            if hi >= self._hi:        # extend to the right: old bins -> left half
                self._hi = self._lo + 2 * width
                merge = lambda c: np.concatenate(          # noqa: E731
                    [c.reshape(-1, 2).sum(axis=1), np.zeros(self.bins // 2, np.int64)])
            else:                     # extend to the left: old bins -> right half
                self._lo = self._hi - 2 * width
                merge = lambda c: np.concatenate(          # noqa: E731
                    [np.zeros(self.bins // 2, np.int64), c.reshape(-1, 2).sum(axis=1)])
            self._total = merge(self._total)
            self._chunks = deque(merge(c) for c in self._chunks)

    def update(self, values):
        """Add a chunk of values (any shape; flattened). Returns self."""
        x = np.asarray(values, dtype=np.float64).ravel()
        nan = np.isnan(x)
        self.n_nan_ += int(nan.sum())
        x = x[~nan]
        if np.isinf(x).any():
            raise ValueError("values contain inf")
        if x.size and self._lo is None:            # first data: set the range
            lo, hi = float(x.min()), float(x.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            pad = (hi - lo) * 1e-9                 # so the maximum lands inside
            self._lo, self._hi = lo, hi + pad
        if x.size and not self.fixed_range:
            self._grow_to(float(x.min()), float(x.max()))
        counts = np.zeros(self.bins, dtype=np.int64)
        if x.size:
            inside = (x >= self._lo) & (x < self._hi) if not self.fixed_range \
                else (x >= self._lo) & (x <= self._hi)
            self.n_outside_ += int((~inside).sum())
            x = x[inside]
            idx = ((x - self._lo) / (self._hi - self._lo) * self.bins).astype(np.intp)
            idx = np.clip(idx, 0, self.bins - 1)
            counts = np.bincount(idx, minlength=self.bins).astype(np.int64)
        if self.window is None:
            self._total += counts
        else:
            self._chunks.append(counts)
            self._total += counts
            if len(self._chunks) > self.window:
                self._total -= self._chunks.popleft()
        return self

    def reset(self):
        """Forget all counts (keeps a fixed range, drops an inferred one)."""
        self.__init__(self.bins, (self._lo, self._hi) if self.fixed_range else None,
                      self.window)
        return self

    @property
    def counts(self):
        """Counts per bin over the window (or the whole stream), shape (bins,)."""
        return self._total.copy()

    @property
    def count(self):
        """Number of values currently in the bins."""
        return int(self._total.sum())

    def quantile(self, q):
        """Estimate quantile(s) ``q`` in [0, 1] by interpolating inside bins.

        The error is at most one bin width. Returns NaN when empty.
        """
        q = np.asarray(q, dtype=np.float64)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("q must be in [0, 1]")
        total = self._total.sum()
        if total == 0:
            return np.full(q.shape, np.nan) if q.ndim else float("nan")
        cdf = np.concatenate([[0], np.cumsum(self._total)]) / total   # at each edge
        # first edge whose CDF reaches q; interpolate within the bin before it
        res = np.interp(q, cdf, self.edges) if np.all(np.diff(cdf) > 0) \
            else self._interp_flat(q, cdf)
        return float(res) if res.ndim == 0 else res

    def _interp_flat(self, q, cdf):
        """np.interp needs increasing x; empty bins make the CDF flat, so step."""
        edges = self.edges
        qs = np.atleast_1d(q)
        j = np.clip(np.searchsorted(cdf, qs, side="left"), 1, self.bins)
        lo_c, hi_c = cdf[j - 1], cdf[j]
        frac = np.divide(qs - lo_c, hi_c - lo_c, out=np.zeros_like(qs), where=hi_c > lo_c)
        res = edges[j - 1] + frac * (edges[j] - edges[j - 1])
        return res.reshape(np.shape(q))


class StreamingStats:
    """Per-feature statistics of a stream of chunks, without storing the data.

    Call :meth:`update_stats` with each chunk of shape (n_samples, n_features).
    Mean and variance are exact (merged with :func:`merge_moments`); quantiles
    are estimated from a :class:`StreamingHistogram` per feature.

    Parameters
    ----------
    bins : int, default 64
        Histogram resolution used for quantiles and ``histogram()``.
    window : int or None
        If set, every statistic covers only the last ``window`` chunks
        (sliding window). Min/max are then recomputed from per-chunk extrema.
    ddof : int, default 0
        Degrees of freedom for ``var`` / ``std`` (1 = sample variance).

    Attributes (after the first chunk)
    ----------------------------------
    n_features_in_, n_chunks_, count (non-NaN per feature), nan_count, mean,
    var, std, min, max.

    NaN values are skipped per feature and counted in ``nan_count``. A feature
    that has only had NaN so far reports mean/var/min/max as NaN.
    """

    def __init__(self, bins=64, window=None, ddof=0):
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0, got {ddof}")
        if window is not None and (not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError(f"window must be None or an int >= 1, got {window!r}")
        self.bins = bins
        self.window = window
        self.ddof = ddof
        self.reset()

    def reset(self):
        """Forget everything seen so far. Returns self."""
        self.n_features_in_ = None
        self.n_chunks_ = 0
        self._n = self._mean = self._m2 = None
        self._min = self._max = None
        self._nan = None
        self._recent = deque()     # per-chunk summaries (window mode)
        self._hists = None
        return self

    def update_stats(self, X_chunk):
        """Fold one chunk into the running statistics and return a summary dict.

        Parameters
        ----------
        X_chunk : array of shape (n_samples, n_features)

        Returns
        -------
        dict : the same as :meth:`summary` after the update.
        """
        from ._validation import check_2d_numeric, check_n_features
        X = check_2d_numeric(X_chunk, "X_chunk")
        check_n_features(X, self.n_features_in_, "X_chunk")
        d = X.shape[1]
        if self.n_features_in_ is None:
            self.n_features_in_ = d
            zeros = np.zeros(d)
            self._n, self._mean, self._m2 = zeros.copy(), zeros.copy(), zeros.copy()
            self._min, self._max = np.full(d, np.inf), np.full(d, -np.inf)
            self._nan = np.zeros(d, dtype=np.int64)
            self._hists = [StreamingHistogram(self.bins, window=self.window) for _ in range(d)]
        if X.shape[0] == 0:
            return self.summary()

        cnt, mu, m2 = chunk_moments(X)
        nan = np.isnan(X).sum(axis=0)
        with np.errstate(all="ignore"):
            cmin = np.where(cnt > 0, np.nanmin(np.where(np.isnan(X), np.inf, X), axis=0), np.inf)
            cmax = np.where(cnt > 0, np.nanmax(np.where(np.isnan(X), -np.inf, X), axis=0), -np.inf)

        if self.window is None:
            self._n, self._mean, self._m2 = merge_moments(self._n, self._mean, self._m2, cnt, mu, m2)
            self._min, self._max = np.minimum(self._min, cmin), np.maximum(self._max, cmax)
            self._nan += nan
        else:
            self._recent.append((cnt, mu, m2, cmin, cmax, nan))
            if len(self._recent) > self.window:
                self._recent.popleft()
            self._recompute_window()
        for j, h in enumerate(self._hists):
            h.update(X[:, j])
        self.n_chunks_ += 1
        return self.summary()

    def _recompute_window(self):
        d = self.n_features_in_
        n, mean, m2 = np.zeros(d), np.zeros(d), np.zeros(d)
        lo, hi = np.full(d, np.inf), np.full(d, -np.inf)
        nan = np.zeros(d, dtype=np.int64)
        for cnt, mu, cm2, cmin, cmax, cnan in self._recent:
            n, mean, m2 = merge_moments(n, mean, m2, cnt, mu, cm2)
            lo, hi = np.minimum(lo, cmin), np.maximum(hi, cmax)
            nan = nan + cnan
        self._n, self._mean, self._m2, self._min, self._max, self._nan = n, mean, m2, lo, hi, nan

    def _require(self):
        if self.n_features_in_ is None:
            raise ValueError("no data seen yet; call update_stats() first")

    @property
    def count(self):
        self._require()
        return self._n.astype(np.int64)

    @property
    def nan_count(self):
        self._require()
        return self._nan.copy()

    @property
    def mean(self):
        self._require()
        return np.where(self._n > 0, self._mean, np.nan)

    @property
    def var(self):
        self._require()
        denom = self._n - self.ddof
        return np.divide(self._m2, denom, out=np.full_like(self._m2, np.nan), where=denom > 0)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def min(self):
        self._require()
        return np.where(np.isfinite(self._min), self._min, np.nan)

    @property
    def max(self):
        self._require()
        return np.where(np.isfinite(self._max), self._max, np.nan)

    def quantile(self, q):
        """Estimated quantile(s) per feature.

        Returns shape (n_features,) for a scalar q, or (len(q), n_features).
        """
        self._require()
        cols = [np.atleast_1d(h.quantile(q)) for h in self._hists]
        out = np.stack(cols, axis=-1)
        return out[0] if np.ndim(q) == 0 else out

    def histogram(self, feature):
        """(counts, edges) of one feature's streaming histogram."""
        self._require()
        h = self._hists[feature]
        return h.counts, h.edges

    def summary(self):
        """Dict of per-feature arrays: count, nan_count, mean, std, min, 25%, 50%, 75%, max."""
        self._require()
        q = self.quantile([0.25, 0.5, 0.75])
        return {
            "count": self.count, "nan_count": self.nan_count, "mean": self.mean,
            "std": self.std, "min": self.min, "25%": q[0], "50%": q[1], "75%": q[2],
            "max": self.max,
        }
