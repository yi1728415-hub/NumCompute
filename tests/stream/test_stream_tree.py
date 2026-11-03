"""DecisionTreeClassifier: correctness, buffered retraining and edge cases."""

import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier, NotFittedError, entropy, gini

# the twelve students from classification-tree.html (submitted, attended, study_group, passed)
STUDENTS = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1],
                     [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 0]])


def blobs(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(int) + (X[:, 4] > 1).astype(int)
    return X, y


# ---------------------------------------------------------------- impurity
def test_gini_and_entropy_known_values():
    assert gini([0, 0, 1, 1]) == pytest.approx(0.5)
    assert entropy([0, 0, 1, 1]) == pytest.approx(1.0)
    assert gini([3, 3, 3]) == 0.0 and entropy([3, 3, 3]) == 0.0
    assert gini([]) == 0.0 and entropy([]) == 0.0


def test_lecture_tree_is_reproduced():
    X, y = STUDENTS[:, :3], STUDENTS[:, 3]
    for crit in ("gini", "entropy"):
        t = DecisionTreeClassifier(max_depth=2, criterion=crit).fit(X, y)
        # root asks "submitted?", its yes side asks "attended?", 3 leaves
        assert t._feature[0] == 0 and t.get_n_leaves() == 3
        assert t.score(X, y) == pytest.approx(11 / 12)      # student 5 stays wrong


def test_deeper_tree_memorises_the_exception():
    X, y = STUDENTS[:, :3], STUDENTS[:, 3]
    assert DecisionTreeClassifier(max_depth=3).fit(X, y).score(X, y) == 1.0


def test_learns_continuous_thresholds():
    X, y = blobs()
    assert DecisionTreeClassifier(max_depth=6).fit(X, y).score(X, y) > 0.95


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_loop_and_vectorised_build_identical_trees(criterion):
    X, y = blobs(300)
    a = DecisionTreeClassifier(max_depth=5, criterion=criterion, split_method="vectorised").fit(X, y)
    b = DecisionTreeClassifier(max_depth=5, criterion=criterion, split_method="loop").fit(X, y)
    assert np.array_equal(a._feature, b._feature)
    assert np.allclose(a._threshold, b._threshold, equal_nan=True)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_max_depth_and_min_samples_split_limit_growth():
    X, y = blobs()
    assert DecisionTreeClassifier(max_depth=0).fit(X, y).get_n_leaves() == 1
    assert DecisionTreeClassifier(max_depth=3).fit(X, y).get_depth() <= 3
    big = DecisionTreeClassifier(max_depth=None, min_samples_split=2).fit(X, y)
    small = DecisionTreeClassifier(max_depth=None, min_samples_split=200).fit(X, y)
    assert small.get_n_leaves() < big.get_n_leaves()


def test_max_features_subsampling_is_reproducible():
    X, y = blobs()
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert DecisionTreeClassifier(max_features=0.5).fit(X, y).score(X, y) > 0.6


def test_predict_proba_rows_sum_to_one_and_match_predict():
    X, y = blobs()
    t = DecisionTreeClassifier(max_depth=4).fit(X, y)
    p = t.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(t.classes_[p.argmax(axis=1)], t.predict(X))


def test_tie_in_leaf_goes_to_smallest_class():
    t = DecisionTreeClassifier(max_depth=0).fit([[0.0], [1.0]], [1, 0])
    assert t.predict([[0.5]])[0] == 0


def test_string_labels_supported():
    X = STUDENTS[:, :3]
    y = np.where(STUDENTS[:, 3] == 1, "pass", "fail")
    t = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert set(t.predict(X)) <= {"pass", "fail"}


# ---------------------------------------------------------------- streaming
def test_buffer_triggers_retrain_and_clears():
    X, y = blobs(400)
    t = DecisionTreeClassifier(max_depth=4, buffer_size=150)
    counts = []
    for s in range(0, 400, 60):
        t.partial_fit(X[s:s + 60], y[s:s + 60])
        counts.append((t.n_retrains_, t.n_buffered_))
    assert counts == [(0, 60), (0, 120), (1, 0), (1, 60), (1, 120), (2, 0), (2, 40)]
    assert [r["n_samples"] for r in t.retrain_log_] == [180, 180]


def test_predict_before_first_retrain_raises():
    t = DecisionTreeClassifier(buffer_size=100)
    t.partial_fit(np.zeros((10, 2)), np.zeros(10))
    with pytest.raises(NotFittedError, match="10"):
        t.predict(np.zeros((1, 2)))


def test_model_is_frozen_between_retrains():
    X, y = blobs(600)
    t = DecisionTreeClassifier(buffer_size=200).partial_fit(X[:200], y[:200])
    before = t.predict(X)
    t.partial_fit(X[200:350], 1 - y[200:350])     # not enough to retrain
    assert np.array_equal(t.predict(X), before)


def test_retrain_uses_only_the_new_buffer():
    X, y = blobs(400)
    t = DecisionTreeClassifier(max_depth=3, buffer_size=200)
    t.partial_fit(X[:200], y[:200])
    t.partial_fit(X[200:], np.zeros(200, int))       # second buffer is all class 0
    assert np.all(t.predict(X) == 0)


def test_flush_trains_on_partial_buffer():
    t = DecisionTreeClassifier(buffer_size=1000).partial_fit(*blobs(50))
    assert not t.is_fitted
    t.flush()
    assert t.is_fitted and t.n_buffered_ == 0
    t.flush()                                        # empty buffer: no-op
    assert t.n_retrains_ == 1


def test_one_big_chunk_equals_fit():
    X, y = blobs(500)
    a = DecisionTreeClassifier(max_depth=5, buffer_size=500).partial_fit(X, y)
    b = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_new_class_mid_stream_extends_classes():
    X, y = blobs(60)
    t = DecisionTreeClassifier(buffer_size=40).partial_fit(X[:40], np.zeros(40, int))
    t.partial_fit(X[40:45], np.full(5, 7))
    assert list(t.classes_) == [0, 7]
    assert t.predict_proba(X[:3]).shape == (3, 2)
    assert np.all(t.predict_proba(X[:3])[:, 1] == 0)   # not learned until next retrain


def test_accept_new_features_zero_pads_old_chunks():
    t = DecisionTreeClassifier(buffer_size=6, accept_new_features=True)
    t.partial_fit(np.eye(2)[[0, 1, 0, 1]], [0, 1, 0, 1])
    t.partial_fit(np.eye(3)[[2, 2]], [1, 1])
    assert t.n_features_in_ == 3 and t.is_fitted
    assert list(t.predict(np.eye(3))) == [0, 1, 1]


def test_new_features_rejected_by_default():
    t = DecisionTreeClassifier(buffer_size=10).partial_fit(np.zeros((3, 2)), [0, 1, 0])
    with pytest.raises(ValueError, match="features"):
        t.partial_fit(np.zeros((3, 3)), [0, 1, 0])


# ---------------------------------------------------------------- edge cases
def test_nan_features_are_mean_filled():
    X, y = blobs()
    X = X.copy()
    X[::5, 0] = np.nan
    X[:, 3] = np.nan                                  # a column that is all NaN
    t = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert t.score(X, y) > 0.9
    assert np.isfinite(t.predict_proba([[np.nan] * 5])).all()


def test_single_class_and_constant_features_give_one_leaf():
    X, _ = blobs(50)
    assert DecisionTreeClassifier().fit(X, np.ones(50, int)).get_n_leaves() == 1
    assert DecisionTreeClassifier().fit(np.ones((10, 3)), [0, 1] * 5).get_n_leaves() == 1


def test_empty_chunk_is_ignored_but_empty_fit_raises():
    t = DecisionTreeClassifier(buffer_size=5).partial_fit(np.empty((0, 2)), [])
    assert t.n_samples_seen_ == 0
    with pytest.raises(ValueError, match="empty"):
        DecisionTreeClassifier().fit(np.empty((0, 2)), [])


@pytest.mark.parametrize("X, y, match", [
    (np.zeros(5), np.zeros(5), "2-D"),
    (np.zeros((5, 2)), np.zeros(4), "samples"),
    (np.zeros((2, 2)), np.zeros((2, 2)), "1-D"),
    ([[1.0, np.inf]], [0], "inf"),
    ([["a", "b"]], [0], "numeric"),
    (np.zeros((2, 1)), [0.0, np.nan], "NaN"),
])
def test_bad_inputs_have_clear_errors(X, y, match):
    with pytest.raises((ValueError, TypeError), match=match):
        DecisionTreeClassifier().fit(X, y)


def test_feature_count_mismatch_at_predict():
    t = DecisionTreeClassifier().fit(*blobs(50))
    with pytest.raises(ValueError, match="5 features"):
        t.predict(np.zeros((2, 3)))


@pytest.mark.parametrize("kwargs", [
    {"max_depth": -1}, {"min_samples_split": 1}, {"criterion": "mse"},
    {"buffer_size": 0}, {"split_method": "fast"}, {"max_features": 0},
    {"max_features": 1.5}, {"max_features": "all"},
])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)
