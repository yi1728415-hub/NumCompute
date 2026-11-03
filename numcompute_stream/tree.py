"""Decision tree classifier with buffered retraining for streaming data.

Streaming strategy (buffered retraining)
----------------------------------------
Each call to :meth:`DecisionTreeClassifier.partial_fit` appends the incoming
chunk to an internal buffer. Once the buffer holds at least ``buffer_size``
samples, the tree is retrained from scratch on the buffered data and the
buffer is cleared. Between retrains, :meth:`predict` uses the most recently
trained tree.

Shapes
------
* ``X``: ``(n_samples, n_features)`` float array (axis 0 = samples).
* ``y``: ``(n_samples,)`` array of class labels (ints or strings).
* ``predict_proba`` returns ``(n_samples, n_classes)``.

Splits are binary threshold questions ``X[:, f] <= t`` (left = yes).
"""

import time

import numpy as np

from .exceptions import NotFittedError

_EPS = 1e-12            # a split must lower impurity by more than this
_CRITERIA = ("gini", "entropy")
_SPLIT_METHODS = ("vectorised", "loop")


# --------------------------------------------------------------------------
# Impurity measures
# --------------------------------------------------------------------------
def _impurity_from_counts(counts, criterion="gini"):
    """Impurity of one or many groups, given their class counts.

    Parameters
    ----------
    counts : array of shape (..., n_classes)
        Class counts; the last axis is the class axis.
    criterion : {"gini", "entropy"}

    Returns
    -------
    ndarray of shape (...)
        0 for a pure (or empty) group; Gini peaks at 1 - 1/k, entropy at log2(k).
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, counts / n, 0.0)
        if criterion == "gini":
            # an empty group has no impurity (without the mask it would give 1)
            return np.where(n[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)   # log2(0) is undefined -> 0
    return -np.sum(p * logp, axis=-1)


def _label_counts(y):
    """Class counts of a label vector (labels encoded as 0..k-1 or any values)."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return np.zeros(0)
    _, counts = np.unique(y, return_counts=True)
    return counts


def gini(y):
    """Gini impurity ``1 - sum(p_c^2)`` of a 1-D label array (0.0 if empty)."""
    return float(_impurity_from_counts(_label_counts(y), "gini"))


def entropy(y):
    """Entropy ``-sum(p_c * log2 p_c)`` in bits of a 1-D label array (0.0 if empty)."""
    return float(_impurity_from_counts(_label_counts(y), "entropy"))


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class DecisionTreeClassifier:
    """Depth-limited CART classification tree supporting buffered streaming.

    Parameters
    ----------
    max_depth : int or None, default=5
        Maximum depth of the tree (root has depth 0). None = grow until pure.
    min_samples_split : int, default=2
        A node with fewer samples than this becomes a leaf.
    max_features : None, int, float, "sqrt" or "log2", default=None
        Number of features examined at each split. None = all features;
        float in (0, 1] = that fraction of the features.
    criterion : {"gini", "entropy"}, default="gini"
    buffer_size : int, default=500
        Number of buffered samples that triggers a retrain in ``partial_fit``.
    split_method : {"vectorised", "loop"}, default="vectorised"
        Implementation of the split search. "loop" is a pure-Python reference
        used for testing and benchmarking; both give the same tree.
    random_state : int or None
        Seed for feature subsampling when ``max_features`` is set.
    accept_new_features : bool, default=False
        If True, a chunk may have *more* columns than earlier ones (as when an
        upstream OneHotEncoder meets a new category). Older buffered chunks
        are zero-padded on the right at retrain time. If False, any change in
        the number of features raises ValueError.

    Attributes
    ----------
    classes_ : ndarray
        Sorted union of all labels seen so far.
    n_features_in_ : int
    n_samples_seen_ : int
        Total samples received through ``fit`` / ``partial_fit``.
    n_buffered_ : int
        Samples currently waiting in the buffer.
    n_retrains_ : int
        Number of times the tree has been (re)trained.
    retrain_log_ : list of dict
        One entry per retrain: samples used, seconds taken, depth, leaves.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 criterion="gini", buffer_size=500, split_method="vectorised",
                 random_state=None, accept_new_features=False):
        self.accept_new_features = accept_new_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.split_method = split_method
        self.random_state = random_state
        self._check_params()
        self._reset_state()

    # ------------------------------------------------------------ set-up
    def _check_params(self):
        md = self.max_depth
        if md is not None and (not isinstance(md, (int, np.integer)) or md < 0):
            raise ValueError(f"max_depth must be None or an int >= 0, got {md!r}")
        mss = self.min_samples_split
        if not isinstance(mss, (int, np.integer)) or mss < 2:
            raise ValueError(f"min_samples_split must be an int >= 2, got {mss!r}")
        if self.criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {self.criterion!r}")
        bs = self.buffer_size
        if not isinstance(bs, (int, np.integer)) or bs < 1:
            raise ValueError(f"buffer_size must be an int >= 1, got {bs!r}")
        if self.split_method not in _SPLIT_METHODS:
            raise ValueError(f"split_method must be one of {_SPLIT_METHODS}, "
                             f"got {self.split_method!r}")
        mf = self.max_features
        ok = (mf is None or mf in ("sqrt", "log2")
              or (isinstance(mf, (int, np.integer)) and not isinstance(mf, bool) and mf >= 1)
              or (isinstance(mf, float) and 0.0 < mf <= 1.0))
        if not ok:
            raise ValueError("max_features must be None, 'sqrt', 'log2', an int >= 1 "
                             f"or a float in (0, 1], got {mf!r}")

    def _reset_state(self):
        """Forget everything learned: tree, classes, buffer and counters."""
        self.classes_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self.n_retrains_ = 0
        self.retrain_log_ = []
        self._buffer_X, self._buffer_y = [], []
        self.n_buffered_ = 0
        self._rng = np.random.default_rng(self.random_state)
        # trained tree (parallel arrays, one entry per node)
        self._feature = None      # split feature, -1 for a leaf
        self._threshold = None    # go left if x[feature] <= threshold
        self._left = None         # child node ids, -1 for a leaf
        self._right = None
        self._value = None        # class probabilities at each node
        self._tree_classes = None  # classes_ at the time the tree was trained
        self._fill_values = None  # per-feature value used in place of NaN
        self._n_tree_features = 0  # columns the current tree was trained on
        self._depth = 0

    @property
    def is_fitted(self):
        """True once at least one tree has been trained."""
        return self._feature is not None

    # ------------------------------------------------------------ validation
    def _validate_X(self, X, fitting):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must be numeric (convertible to a float array); "
                            "encode categorical columns first, e.g. with OneHotEncoder") from exc
        if X.ndim == 1:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got 1-D shape {X.shape}. "
                             "Use X.reshape(-1, 1) for a single feature "
                             "or X.reshape(1, -1) for a single sample.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
        if np.isinf(X).any():
            raise ValueError("X contains inf values; only finite numbers or NaN are allowed")
        if self.n_features_in_ is None:
            if fitting:
                self.n_features_in_ = X.shape[1]
        elif fitting and self.accept_new_features and X.shape[1] > self.n_features_in_:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but this tree was given "
                             f"{self.n_features_in_} features before")
        return X

    @staticmethod
    def _validate_y(y, n_samples):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D (n_samples,), got shape {y.shape}")
        if y.shape[0] != n_samples:
            raise ValueError(f"X has {n_samples} samples but y has {y.shape[0]}")
        if y.dtype.kind == "f" and np.isnan(y).any():
            raise ValueError("y contains NaN labels")
        return y

    def _update_classes(self, y):
        try:
            new = np.unique(y)
            self.classes_ = new if self.classes_ is None else np.union1d(self.classes_, new)
        except TypeError as exc:
            raise TypeError("labels in y must all be of one comparable type "
                            "(e.g. all ints or all strings)") from exc

    # ------------------------------------------------------------ public API
    def fit(self, X, y):
        """Train a tree from scratch on (X, y), discarding any previous state.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._reset_state()
        X = self._validate_X(X, fitting=True)
        y = self._validate_y(y, X.shape[0])
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset (0 samples)")
        self._update_classes(y)
        self.n_samples_seen_ = X.shape[0]
        self._fit_tree(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Add a chunk to the buffer; retrain once ``buffer_size`` is reached.

        An empty chunk is ignored. Returns self.
        """
        X = self._validate_X(X_chunk, fitting=True)
        y = self._validate_y(y_chunk, X.shape[0])
        if X.shape[0] == 0:
            return self
        self._update_classes(y)
        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        self.n_buffered_ += X.shape[0]
        self.n_samples_seen_ += X.shape[0]
        if self.n_buffered_ >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self):
        """Retrain on whatever is in the buffer now, even if below buffer_size.

        Useful at the end of a stream. Does nothing if the buffer is empty.
        """
        if self.n_buffered_ > 0:
            self._retrain_from_buffer()
        return self

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, len(classes_))."""
        self._check_fitted()
        X = self._validate_X(X, fitting=False)
        leaves = self._apply(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        # classes seen after the last retrain get probability 0 until next retrain
        cols = np.searchsorted(self.classes_, self._tree_classes)
        out[:, cols] = self._value[leaves]
        return out

    def predict(self, X):
        """Predicted labels, shape (n_samples,). Ties go to the smallest class."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """Accuracy of predict(X) against y."""
        y_pred = self.predict(X)
        y = self._validate_y(y, y_pred.shape[0])
        if y.size == 0:
            return float("nan")
        return float(np.mean(y_pred == y))

    def get_depth(self):
        self._check_fitted()
        return self._depth

    def get_n_leaves(self):
        self._check_fitted()
        return int(np.sum(self._feature < 0))

    def __repr__(self):
        return (f"DecisionTreeClassifier(max_depth={self.max_depth}, "
                f"min_samples_split={self.min_samples_split}, "
                f"max_features={self.max_features!r}, criterion={self.criterion!r}, "
                f"buffer_size={self.buffer_size}, split_method={self.split_method!r})")

    # ------------------------------------------------------------ internals
    def _check_fitted(self):
        if not self.is_fitted:
            raise NotFittedError(
                "This DecisionTreeClassifier has not been trained yet: call fit(), or keep "
                f"calling partial_fit() until {self.buffer_size} samples are buffered "
                f"(currently {self.n_buffered_}), or call flush().")

    def _retrain_from_buffer(self):
        d = self.n_features_in_
        X = np.vstack([np.pad(c, ((0, 0), (0, d - c.shape[1]))) for c in self._buffer_X])
        y = np.concatenate(self._buffer_y)
        self._buffer_X, self._buffer_y = [], []
        self.n_buffered_ = 0
        self._fit_tree(X, y)

    def _fit_tree(self, X, y):
        """Build a brand-new tree on (X, y) and log the retrain."""
        start = time.perf_counter()

        # NaN handling: remember each column's mean and use it in place of NaN,
        # both now and at prediction time. An all-NaN column is filled with 0.
        nan_mask = np.isnan(X)
        if nan_mask.any():
            counts = (~nan_mask).sum(axis=0)
            sums = np.where(nan_mask, 0.0, X).sum(axis=0)
            fill = np.divide(sums, counts, out=np.zeros(X.shape[1]), where=counts > 0)
            X = np.where(nan_mask, fill, X)
        else:
            fill = np.zeros(X.shape[1])
        self._fill_values = fill
        self._n_tree_features = X.shape[1]

        self._tree_classes = self.classes_.copy()
        y_enc = np.searchsorted(self._tree_classes, y)
        self._build(X, y_enc, len(self._tree_classes))

        self.n_retrains_ += 1
        self.retrain_log_.append({
            "retrain": self.n_retrains_,
            "n_samples": int(X.shape[0]),
            "seconds": time.perf_counter() - start,
            "depth": self._depth,
            "n_leaves": int(np.sum(self._feature < 0)),
        })

    def _build(self, X, y, k):
        """Grow the tree depth-first with an explicit stack (no recursion limit)."""
        feature, threshold, left, right, value = [], [], [], [], []
        max_depth_seen = 0
        # each stack item: (row indices, depth, parent id, is_left_child)
        stack = [(np.arange(X.shape[0]), 0, -1, False)]
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            counts = np.bincount(y[idx], minlength=k).astype(float)
            n = idx.size
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts / n)
            max_depth_seen = max(max_depth_seen, depth)

            # stop rules, checked before searching
            if ((self.max_depth is not None and depth >= self.max_depth)
                    or n < self.min_samples_split
                    or np.count_nonzero(counts) <= 1):
                continue
            parent_imp = float(_impurity_from_counts(counts, self.criterion))
            best = self._best_split(X[idx], y[idx], counts, parent_imp)
            if best is None:                     # no question lowers impurity
                continue
            f, t, _gain = best
            feature[node], threshold[node] = f, t
            go_left = X[idx, f] <= t
            # push right first so the left child is built first
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        self._feature = np.array(feature, dtype=int)
        self._threshold = np.array(threshold, dtype=float)
        self._left = np.array(left, dtype=int)
        self._right = np.array(right, dtype=int)
        self._value = np.array(value, dtype=float)
        self._depth = max_depth_seen

    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            m = int(np.sqrt(d))
        elif mf == "log2":
            m = int(np.log2(d)) if d > 1 else 1
        elif isinstance(mf, float):
            m = int(mf * d)
        else:
            m = int(mf)
        return int(np.clip(m, 1, d))

    def _choose_features(self, d):
        m = self._n_features_to_try(d)
        if m >= d:
            return np.arange(d)
        return np.sort(self._rng.choice(d, size=m, replace=False))

    def _best_split(self, X, y, counts, parent_imp):
        feats = self._choose_features(X.shape[1])
        if self.split_method == "loop":
            return self._best_split_loop(X, y, counts, parent_imp, feats)
        return self._best_split_vectorised(X, y, counts, parent_imp, feats)

    @staticmethod
    def _midpoint(a, b):
        """Threshold between two sorted values a < b, robust to rounding/overflow."""
        t = a / 2.0 + b / 2.0
        return t if a <= t < b else a

    def _best_split_vectorised(self, X, y, counts, parent_imp, feats):
        """Score every (feature, threshold) pair at once with NumPy.

        Sort each feature column, then a cumulative sum of one-hot labels gives
        the class counts left of every possible cut in one operation.
        Ties: the smallest feature index wins, then the smallest threshold.
        """
        n = X.shape[0]
        k = counts.size
        Xf = X[:, feats]                                   # (n, m)
        order = np.argsort(Xf, axis=0, kind="stable")      # (n, m)
        Xs = np.take_along_axis(Xf, order, axis=0)         # sorted values
        onehot = np.eye(k)[y]                              # (n, k)
        left = np.cumsum(onehot[order], axis=0)[:-1]       # (n-1, m, k)
        right = counts - left
        n_left = np.arange(1, n, dtype=float)[:, None]     # (n-1, 1)
        n_right = n - n_left
        child_imp = (n_left * _impurity_from_counts(left, self.criterion)
                     + n_right * _impurity_from_counts(right, self.criterion)) / n
        valid = Xs[1:] > Xs[:-1]       # cannot cut between two equal values
        gain = np.where(valid, parent_imp - child_imp, -np.inf)   # (n-1, m)

        flat = np.argmax(gain.T)       # row-major over (feature, threshold)
        j, i = divmod(int(flat), n - 1)
        best_gain = gain[i, j]
        if not best_gain > _EPS:
            return None
        return int(feats[j]), float(self._midpoint(Xs[i, j], Xs[i + 1, j])), float(best_gain)

    def _best_split_loop(self, X, y, counts, parent_imp, feats):
        """Reference implementation with Python loops (for tests and benchmarks)."""
        n = X.shape[0]
        k = counts.size
        best = None
        best_gain = _EPS
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            left = np.zeros(k)
            for i in range(n - 1):
                left[ys[i]] += 1
                if not xs[i + 1] > xs[i]:
                    continue
                n_left = i + 1
                right = counts - left
                child_imp = (n_left * _impurity_from_counts(left, self.criterion)
                             + (n - n_left) * _impurity_from_counts(right, self.criterion)) / n
                g = parent_imp - child_imp
                if g > best_gain:
                    best_gain = g
                    best = (int(f), float(self._midpoint(xs[i], xs[i + 1])), float(g))
        return best

    def _apply(self, X):
        """Leaf id reached by each row; all rows descend one level per step."""
        if X.shape[0] == 0:
            return np.zeros(0, dtype=int)
        X = X[:, :self._n_tree_features]          # columns added after the last retrain
        X = np.where(np.isnan(X), self._fill_values, X)
        rows = np.arange(X.shape[0])
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(self._depth):
            f = self._feature[node]
            internal = f >= 0
            if not internal.any():
                break
            go_left = X[rows, np.where(internal, f, 0)] <= self._threshold[node]
            child = np.where(go_left, self._left[node], self._right[node])
            node = np.where(internal, child, node)
        return node
