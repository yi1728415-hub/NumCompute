"""Streaming preprocessing and statistics must match their batch versions."""

import numpy as np
import pytest

from numcompute_stream import (Imputer, MinMaxScaler, NotFittedError, OneHotEncoder,
                               StandardScaler, StreamingHistogram, StreamingStats, merge_moments)
from numcompute_stream.stats import chunk_moments


def data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(3, 4, (n, 4))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


# ---------------------------------------------------------------- moments
def test_merge_moments_matches_numpy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=37), rng.normal(5, 2, size=81)
    n, mu, m2 = merge_moments(len(a), a.mean(), ((a - a.mean()) ** 2).sum(),
                              len(b), b.mean(), ((b - b.mean()) ** 2).sum())
    both = np.concatenate([a, b])
    assert n == 118 and mu == pytest.approx(both.mean()) and m2 / n == pytest.approx(both.var())


def test_merge_with_empty_group_is_identity():
    n, mu, m2 = merge_moments(0, 0, 0, 4, 2.5, 5.0)
    assert (n, mu, m2) == (4, 2.5, 5.0)


def test_chunk_moments_ignores_nan():
    cnt, mu, m2 = chunk_moments([[1.0, np.nan], [3.0, np.nan]])
    assert list(cnt) == [2, 0] and mu[0] == 2 and m2[0] == 2 and mu[1] == 0


# ---------------------------------------------------------------- scalers
@pytest.mark.parametrize("n_chunks", [1, 3, 17])
def test_standard_scaler_partial_fit_equals_batch(n_chunks):
    X = data()
    s = StandardScaler()
    for c in np.array_split(X, n_chunks):
        s.partial_fit(c)
    assert np.allclose(s.mean_, np.nanmean(X, axis=0))
    assert np.allclose(s.var_, np.nanvar(X, axis=0))
    Z = s.transform(X)
    assert np.allclose(np.nanmean(Z, axis=0), 0, atol=1e-12)
    assert np.allclose(np.nanstd(Z, axis=0), 1)


def test_standard_scaler_is_stable_with_large_offset():
    rng = np.random.default_rng(2)
    X = 1e9 + rng.normal(size=(5000, 1))
    s = StandardScaler()
    for c in np.array_split(X, 50):
        s.partial_fit(c)
    assert s.var_[0] == pytest.approx(np.var(X), rel=1e-6)


def test_zero_variance_feature_gets_unit_scale():
    s = StandardScaler().partial_fit(np.full((5, 1), 7.0))
    assert s.scale_[0] == 1.0 and np.all(s.transform([[7.0], [8.0]]).ravel() == [0, 1])


def test_scaler_round_trip_and_not_fitted_error():
    X = data(50)
    s = StandardScaler().fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X, equal_nan=True)
    with pytest.raises(NotFittedError):
        StandardScaler().transform(X)


def test_fit_forgets_previous_chunks():
    s = StandardScaler().partial_fit([[100.0]]).fit([[1.0], [3.0]])
    assert s.mean_[0] == 2.0


def test_minmax_scaler_tracks_running_extremes():
    m = MinMaxScaler().partial_fit([[0.0], [5.0]]).partial_fit([[10.0]])
    assert m.transform([[5.0]])[0, 0] == 0.5
    assert MinMaxScaler(clip=True).fit([[0.0], [1.0]]).transform([[3.0]])[0, 0] == 1.0


# ---------------------------------------------------------------- imputer
def test_mean_imputer_updates_on_the_fly():
    imp = Imputer().partial_fit([[1.0, np.nan]])
    assert list(imp.statistics_) == [1.0, 0.0]            # all-NaN so far -> fill_value
    imp.partial_fit([[3.0, 4.0]])
    assert list(imp.transform([[np.nan, np.nan]])[0]) == [2.0, 4.0]


def test_most_frequent_and_constant_imputer():
    mf = Imputer("most_frequent").partial_fit([[1.0], [2.0]]).partial_fit([[2.0], [np.nan]])
    assert mf.statistics_[0] == 2.0
    const = Imputer("constant", fill_value=-1).fit([[np.nan]])
    assert const.transform([[np.nan]])[0, 0] == -1


# ---------------------------------------------------------------- one-hot
def test_one_hot_expands_incrementally_and_keeps_old_columns():
    e = OneHotEncoder().partial_fit([["a", "x"], ["b", "x"]])
    before = e.transform([["a", "x"]])
    e.partial_fit([["c", "y"]])
    after = e.transform([["a", "x"]])
    assert after.shape[1] == before.shape[1] + 2
    assert np.array_equal(after[:, :before.shape[1]], before)     # old columns unchanged
    assert e.get_feature_names() == ["x0=a", "x0=b", "x1=x", "x0=c", "x1=y"]


def test_one_hot_unknown_and_missing_values():
    e = OneHotEncoder().fit([["red"], [None]])
    assert list(e.categories_[0]) == ["red", "nan"]
    assert e.transform([["blue"]]).sum() == 0
    with pytest.raises(ValueError, match="unknown"):
        OneHotEncoder(handle_unknown="error").fit([["red"]]).transform([["blue"]])


def test_one_hot_max_categories_bounds_width():
    e = OneHotEncoder(max_categories=2).fit([["a"], ["b"], ["c"]])
    assert e.n_output_features_ == 2


# ---------------------------------------------------------------- histogram / stats
def test_streaming_histogram_grows_without_losing_counts():
    h = StreamingHistogram(bins=8)
    h.update([0, 1, 2]).update([10]).update([-30])
    assert h.count == 5 and h.edges[0] <= -30 and h.edges[-1] > 10


def test_streaming_histogram_sliding_window():
    h = StreamingHistogram(bins=4, range=(0, 4), window=2)
    h.update([0.5]).update([1.5]).update([2.5])
    assert h.count == 2 and list(h.counts) == [0, 1, 1, 0]


def test_streaming_quantile_close_to_exact():
    x = np.random.default_rng(3).normal(size=20000)
    h = StreamingHistogram(bins=256)
    for c in np.array_split(x, 40):
        h.update(c)
    width = h.edges[1] - h.edges[0]
    assert abs(h.quantile(0.5) - np.median(x)) <= width
    assert np.isnan(StreamingHistogram().quantile(0.5))


def test_streaming_stats_equal_batch_and_count_nan():
    X = data()
    s = StreamingStats(ddof=1)
    for c in np.array_split(X, 13):
        out = s.update_stats(c)
    assert np.allclose(out["mean"], np.nanmean(X, axis=0))
    assert np.allclose(s.var, np.nanvar(X, axis=0, ddof=1))
    assert np.array_equal(s.nan_count, np.isnan(X).sum(axis=0))
    assert np.allclose(s.min, np.nanmin(X, axis=0)) and np.allclose(s.max, np.nanmax(X, axis=0))


def test_streaming_stats_window_covers_recent_chunks_only():
    X = data()
    chunks = np.array_split(X, 10)
    s = StreamingStats(window=3)
    for c in chunks:
        s.update_stats(c)
    assert np.allclose(s.mean, np.nanmean(np.vstack(chunks[-3:]), axis=0))


def test_streaming_stats_errors_and_all_nan_feature():
    s = StreamingStats()
    with pytest.raises(ValueError, match="no data"):
        _ = s.mean
    s.update_stats([[1.0, np.nan], [2.0, np.nan]])
    assert np.isnan(s.mean[1]) and s.count[1] == 0
    with pytest.raises(ValueError, match="features"):
        s.update_stats(np.zeros((2, 3)))
