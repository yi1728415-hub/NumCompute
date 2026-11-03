"""Streaming-capable preprocessing: scalers, imputer and one-hot encoder.

Every transformer has the same API:

``fit(X)``          learn from X only (forgets earlier state), returns self
``partial_fit(X)``  fold one more chunk into the learned state, returns self
``transform(X)``    apply what has been learned so far, never changes state
``fit_transform(X)``

Shapes: X is (n_samples, n_features), axis 0 = samples. Statistics are kept
per feature, NaN is ignored when learning, and an empty chunk passed to
``partial_fit`` is a no-op.
"""

import numpy as np

from ._validation import check_2d_numeric, check_n_features
from .exceptions import NotFittedError
from .stats import chunk_moments, merge_moments

__all__ = ["StandardScaler", "MinMaxScaler", "Imputer", "SimpleImputer", "OneHotEncoder"]


class _StreamingTransformer:
    """Shared plumbing: feature-count checks and the fit/partial_fit relation."""

    def _reset(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y=None):
        """Forget earlier chunks and learn from X alone. Returns self."""
        self._reset()
        return self.partial_fit(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def _start_chunk(self, X):
        X = check_2d_numeric(X)
        check_n_features(X, self.n_features_in_)
        if self.n_features_in_ is None and X.shape[0] > 0:
            self.n_features_in_ = X.shape[1]
        return X

    def _check_transform_input(self, X):
        if self.n_features_in_ is None:
            raise NotFittedError(f"{type(self).__name__} has not seen any data; "
                                 "call fit() or partial_fit() first")
        X = check_2d_numeric(X)
        check_n_features(X, self.n_features_in_)
        return X


class StandardScaler(_StreamingTransformer):
    """Standardise features to zero mean and unit variance, chunk by chunk.

    ``partial_fit`` keeps a running per-feature count, mean and M2 (sum of
    squared deviations) and merges each chunk with Chan's parallel version of
    Welford's algorithm (:func:`numcompute_stream.stats.merge_moments`). The
    result is identical to fitting on all chunks at once, up to rounding.

    Parameters
    ----------
    with_mean, with_std : bool, default True
    eps : float, default 1e-12
        A feature whose standard deviation is <= eps (zero-variance chunk or
        constant feature) gets scale 1, so it is centred but not blown up.

    Attributes
    ----------
    mean_, var_, scale_ : ndarray (n_features,)
    n_samples_seen_ : ndarray (n_features,) of non-NaN counts
    """

    def __init__(self, with_mean=True, with_std=True, eps=1e-12):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.mean_ = self.var_ = self.scale_ = None
        self._m2 = None

    def partial_fit(self, X, y=None):
        """Update running mean/variance with one chunk. Returns self."""
        X = self._start_chunk(X)
        if X.shape[0] == 0:
            return self
        cnt, mu, m2 = chunk_moments(X)
        if self.mean_ is None:
            self.n_samples_seen_, self.mean_, self._m2 = cnt, mu, m2
        else:
            self.n_samples_seen_, self.mean_, self._m2 = merge_moments(
                self.n_samples_seen_, self.mean_, self._m2, cnt, mu, m2)
        n = self.n_samples_seen_
        self.var_ = np.divide(self._m2, n, out=np.zeros_like(self._m2), where=n > 0)
        std = np.sqrt(self.var_)
        self.scale_ = np.where(std > self.eps, std, 1.0)
        return self

    def transform(self, X):
        """Return (X - mean_) / scale_; NaN stays NaN. Shape is unchanged."""
        X = self._check_transform_input(X)
        out = X - self.mean_ if self.with_mean else X.copy()
        return out / self.scale_ if self.with_std else out

    def inverse_transform(self, X):
        X = self._check_transform_input(X)
        out = X * self.scale_ if self.with_std else X.copy()
        return out + self.mean_ if self.with_mean else out


class MinMaxScaler(_StreamingTransformer):
    """Scale features into ``feature_range`` using the running min and max.

    Values beyond the extremes seen so far map outside the range; set
    ``clip=True`` to clamp them. A constant feature uses a range of 1.
    """

    def __init__(self, feature_range=(0.0, 1.0), clip=False):
        lo, hi = feature_range
        if not lo < hi:
            raise ValueError(f"feature_range must satisfy min < max, got {feature_range}")
        self.feature_range = (float(lo), float(hi))
        self.clip = clip
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.data_min_ = self.data_max_ = self.data_range_ = None

    def partial_fit(self, X, y=None):
        X = self._start_chunk(X)
        if X.shape[0] == 0:
            return self
        with np.errstate(all="ignore"):
            lo = np.min(np.where(np.isnan(X), np.inf, X), axis=0)
            hi = np.max(np.where(np.isnan(X), -np.inf, X), axis=0)
        if self.data_min_ is None:
            self.data_min_, self.data_max_ = lo, hi
        else:
            self.data_min_ = np.minimum(self.data_min_, lo)
            self.data_max_ = np.maximum(self.data_max_, hi)
        rng = self.data_max_ - self.data_min_
        self.data_range_ = np.where(np.isfinite(rng) & (rng > 0), rng, 1.0)
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        lo, hi = self.feature_range
        base = np.where(np.isfinite(self.data_min_), self.data_min_, 0.0)
        out = (X - base) / self.data_range_ * (hi - lo) + lo
        return np.clip(out, lo, hi) if self.clip else out


class Imputer(_StreamingTransformer):
    """Fill NaNs with a per-feature value that is updated on the fly.

    Parameters
    ----------
    strategy : {"mean", "most_frequent", "constant"}, default "mean"
        "mean": running mean of the non-NaN values seen so far.
        "most_frequent": running mode (value counts are merged per chunk;
        ties go to the smallest value).
        "constant": always ``fill_value``.
    fill_value : float, default 0.0
        Used by "constant", and for any feature that has had only NaN so far.

    Attributes
    ----------
    statistics_ : ndarray (n_features,)
        The value currently used for each feature.
    """

    _STRATEGIES = ("mean", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.statistics_ = None
        self._n = self._mean = None
        self._value_counts = None      # list of (values, counts) per feature

    def partial_fit(self, X, y=None):
        X = self._start_chunk(X)
        if X.shape[0] == 0:
            return self
        d = X.shape[1]
        if self.strategy == "mean":
            cnt, mu, _ = chunk_moments(X)
            if self._n is None:
                self._n, self._mean = cnt, mu
            else:
                self._n, self._mean, _ = merge_moments(self._n, self._mean, 0, cnt, mu, 0)
            stat = np.where(self._n > 0, self._mean, self.fill_value)
        elif self.strategy == "most_frequent":
            if self._value_counts is None:
                self._value_counts = [(np.empty(0), np.empty(0, np.int64)) for _ in range(d)]
            stat = np.full(d, self.fill_value)
            for j in range(d):           # one np.unique per feature, not per value
                col = X[:, j]
                vals, cnts = np.unique(col[~np.isnan(col)], return_counts=True)
                old_v, old_c = self._value_counts[j]
                allv = np.concatenate([old_v, vals])
                allc = np.concatenate([old_c, cnts])
                merged_v, inv = np.unique(allv, return_inverse=True)
                merged_c = np.bincount(inv, weights=allc).astype(np.int64)
                self._value_counts[j] = (merged_v, merged_c)
                if merged_v.size:
                    stat[j] = merged_v[np.argmax(merged_c)]   # first max = smallest value
        else:
            stat = np.full(d, self.fill_value)
        self.statistics_ = stat
        return self

    def transform(self, X):
        """Return a copy of X with NaN replaced by ``statistics_``."""
        X = self._check_transform_input(X)
        return np.where(np.isnan(X), self.statistics_, X)


# Name used by the Assignment 2.1 code base.
SimpleImputer = Imputer


class OneHotEncoder:
    """One-hot encode categorical columns, adding new categories as they appear.

    Parameters
    ----------
    handle_unknown : {"ignore", "error"}, default "ignore"
        What ``transform`` does with a category not learned yet: an all-zero
        block ("ignore") or ValueError ("error").
    max_categories : int or None
        Stop learning new categories for a feature once it has this many
        (later new values are treated as unknown). Bounds the output width.

    Notes
    -----
    * Values are compared as strings, so the column may hold text or numbers;
      NaN / None / "" become the category ``"nan"``.
    * Output columns are in the order the categories were **learned**, across
      all features: a category met for the first time always becomes a new
      column at the far right. Existing output columns therefore never move
      or change meaning when the encoder grows. The output width can grow
      between chunks, so a downstream model must accept new columns
      (``DecisionTreeClassifier(accept_new_features=True)`` zero-pads older
      buffered chunks, which is exactly what their one-hot encoding would be).

    Attributes
    ----------
    categories_ : list of ndarray, one per input feature
    n_output_features_ : int
    """

    def __init__(self, handle_unknown="ignore", max_categories=None):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        if max_categories is not None and (not isinstance(max_categories, (int, np.integer))
                                           or max_categories < 1):
            raise ValueError("max_categories must be None or an int >= 1")
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.categories_ = None
        self._col_order = []        # (feature, index within categories_[feature])

    @staticmethod
    def _as_str_2d(X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {arr.shape}")
        out = arr.astype(str)
        missing = np.vectorize(lambda v: v is None or v == "" or
                               (isinstance(v, float) and np.isnan(v)), otypes=[bool])(arr) \
            if arr.size else np.zeros(arr.shape, bool)
        out[missing] = "nan"
        return out

    @property
    def n_output_features_(self):
        return 0 if self.categories_ is None else int(sum(len(c) for c in self.categories_))

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Append categories that have not been seen before. Returns self."""
        S = self._as_str_2d(X)
        if self.n_features_in_ is not None and S.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {S.shape[1]} features, but {self.n_features_in_} "
                             "were seen in earlier calls")
        if S.shape[0] == 0:
            return self
        if self.categories_ is None:
            self.n_features_in_ = S.shape[1]
            self.categories_ = [np.empty(0, dtype=object) for _ in range(S.shape[1])]
        for j in range(S.shape[1]):
            vals, first = np.unique(S[:, j], return_index=True)
            vals = vals[np.argsort(first)]                     # first-seen order
            new = vals[~np.isin(vals, self.categories_[j])]
            if self.max_categories is not None:
                new = new[:max(0, self.max_categories - len(self.categories_[j]))]
            if new.size:
                start = len(self.categories_[j])
                self.categories_[j] = np.concatenate([self.categories_[j], new.astype(object)])
                self._col_order.extend((j, start + i) for i in range(new.size))
        return self

    def transform(self, X):
        """Return an int array of shape (n_samples, n_output_features_)."""
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder has not seen any data; call fit() or "
                                 "partial_fit() first")
        S = self._as_str_2d(X)
        if S.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {S.shape[1]} features, but {self.n_features_in_} "
                             "were seen in earlier calls")
        blocks = []
        for j, cats in enumerate(self.categories_):
            match = S[:, j][:, None] == cats.astype(str)[None, :]      # (n, n_cats)
            if self.handle_unknown == "error" and not match.any(axis=1).all():
                unknown = np.unique(S[~match.any(axis=1), j])
                raise ValueError(f"unknown categories in feature {j}: {list(unknown)}")
            blocks.append(match.astype(np.int64))
        if not blocks:
            return np.zeros((S.shape[0], 0), np.int64)
        grouped = np.concatenate(blocks, axis=1)          # feature-by-feature
        offsets = np.cumsum([0] + [len(c) for c in self.categories_])
        perm = [offsets[j] + i for j, i in self._col_order]
        return grouped[:, perm]                           # learning order

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names(self, input_names=None):
        """Names like ``"colour=red"`` for every output column."""
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder has not seen any data")
        names = input_names or [f"x{j}" for j in range(self.n_features_in_)]
        return [f"{names[j]}={self.categories_[j][i]}" for j, i in self._col_order]
