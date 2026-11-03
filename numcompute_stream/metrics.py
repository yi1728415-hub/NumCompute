"""Evaluation metrics implemented with NumPy only.

Two layers:

* **Batch functions** (from Assignment 2.1): ``accuracy_score``,
  ``precision_score``, ``recall_score``, ``f1_score``, ``confusion_matrix``,
  ``roc_curve``, ``roc_auc_score``, ``mean_squared_error``.
* **Streaming metrics** (Assignment 2.2): objects with ``update(y_true_chunk,
  y_pred_chunk)``, ``result()`` and ``reset()`` that accumulate over a stream,
  optionally over a rolling window of the last ``window`` chunks:
  ``StreamingConfusionMatrix``, ``StreamingAccuracy``, ``StreamingPrecision``,
  ``StreamingRecall``, ``StreamingF1``, ``StreamingAUC``.

The public API intentionally exposes both concise names (``accuracy``) and
scikit-learn-style names (``accuracy_score``) so that the module is pleasant
to use in teaching examples while remaining familiar to Python users.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Optional

import numpy as np

Average = Optional[Literal["binary", "micro", "macro", "weighted"]]


def _as_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_classification_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true.shape} and {pred.shape}"
        )
    return true, pred


def _unique_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    combined = np.concatenate((y_true, y_pred))
    try:
        return np.unique(combined)
    except TypeError:
        # Mixed, non-orderable object labels still receive a deterministic
        # first-seen order.
        unique: list[Any] = []
        for value in combined.tolist():
            if not any(value == existing for existing in unique):
                unique.append(value)
        return np.asarray(unique, dtype=object)


def _validate_labels(
    labels: Iterable[Any] | None, y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    if labels is None:
        return _unique_labels(y_true, y_pred)

    result = _as_1d(list(labels), "labels")
    if len(_unique_labels(result, result)) != len(result):
        raise ValueError("labels must not contain duplicates")

    observed = _unique_labels(y_true, y_pred)
    for value in observed:
        if not np.any(result == value):
            raise ValueError(f"observed label {value!r} is missing from labels")
    return result


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Return the fraction of exactly matching predictions."""

    true, pred = _validate_classification_inputs(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Iterable[Any] | None = None,
    normalize: Literal["true", "pred", "all"] | None = None,
) -> np.ndarray:
    """Build a confusion matrix with true labels on rows and predictions on columns.

    Parameters
    ----------
    normalize:
        ``"true"`` normalizes rows, ``"pred"`` normalizes columns and
        ``"all"`` normalizes the complete matrix. Empty rows/columns remain 0.
    """

    true, pred = _validate_classification_inputs(y_true, y_pred)
    class_labels = _validate_labels(labels, true, pred)
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)

    for row, label_true in enumerate(class_labels):
        true_mask = true == label_true
        for column, label_pred in enumerate(class_labels):
            matrix[row, column] = np.count_nonzero(true_mask & (pred == label_pred))

    if normalize is None:
        return matrix
    if normalize not in {"true", "pred", "all"}:
        raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'")

    result = matrix.astype(float)
    if normalize == "true":
        denominator = result.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denominator = result.sum(axis=0, keepdims=True)
    else:
        denominator = np.asarray(result.sum())
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator != 0)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, zero_division: int) -> np.ndarray:
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")
    default = np.full_like(numerator, float(zero_division), dtype=float)
    return np.divide(numerator, denominator, out=default, where=denominator != 0)


def _classification_statistic(
    y_true: Any,
    y_pred: Any,
    *,
    metric: Literal["precision", "recall", "f1"],
    average: Average,
    labels: Iterable[Any] | None,
    pos_label: Any,
    zero_division: int,
) -> float | np.ndarray:
    true, pred = _validate_classification_inputs(y_true, y_pred)
    class_labels = _validate_labels(labels, true, pred)
    if average == "binary" and not np.any(class_labels == pos_label):
        if labels is not None:
            raise ValueError(f"pos_label={pos_label!r} is not present in labels")
        if len(class_labels) > 1:
            raise ValueError(f"pos_label={pos_label!r} is not a valid label")
        class_labels = np.append(class_labels, pos_label)
    matrix = confusion_matrix(true, pred, labels=class_labels)

    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    support = matrix.sum(axis=1).astype(float)

    precision_values = _safe_divide(tp, tp + fp, zero_division)
    recall_values = _safe_divide(tp, tp + fn, zero_division)
    if metric == "precision":
        values = precision_values
    elif metric == "recall":
        values = recall_values
    else:
        values = _safe_divide(
            2.0 * precision_values * recall_values,
            precision_values + recall_values,
            zero_division,
        )

    if average is None:
        return values
    if average == "binary":
        if len(class_labels) > 2:
            raise ValueError("average='binary' is only valid for binary targets")
        positions = np.flatnonzero(class_labels == pos_label)
        if positions.size == 0:
            raise ValueError(f"pos_label={pos_label!r} is not a valid label")
        return float(values[int(positions[0])])
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        total = support.sum()
        if total == 0:
            return float(zero_division)
        return float(np.average(values, weights=support))
    if average == "micro":
        total_tp = np.asarray([tp.sum()])
        if metric == "precision":
            return float(_safe_divide(total_tp, np.asarray([tp.sum() + fp.sum()]), zero_division)[0])
        if metric == "recall":
            return float(_safe_divide(total_tp, np.asarray([tp.sum() + fn.sum()]), zero_division)[0])
        micro_precision = _safe_divide(
            total_tp, np.asarray([tp.sum() + fp.sum()]), zero_division
        )
        micro_recall = _safe_divide(
            total_tp, np.asarray([tp.sum() + fn.sum()]), zero_division
        )
        return float(
            _safe_divide(
                2.0 * micro_precision * micro_recall,
                micro_precision + micro_recall,
                zero_division,
            )[0]
        )
    raise ValueError("average must be one of 'binary', 'micro', 'macro', 'weighted', or None")


def precision_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute binary or multiclass precision."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="precision",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def recall_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute binary or multiclass recall."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="recall",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute the harmonic mean of precision and recall."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="f1",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Return the mean squared difference between numeric arrays."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true.shape} and {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must contain only finite values")
    return float(np.mean(np.square(true - pred)))


def roc_curve(
    y_true: Any,
    y_score: Any,
    *,
    pos_label: Any = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute false-positive and true-positive rates for binary scores."""

    true = _as_1d(y_true, "y_true")
    score = np.asarray(y_score, dtype=float)
    if score.ndim != 1:
        raise ValueError("y_score must be one-dimensional")
    if true.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if not np.all(np.isfinite(score)):
        raise ValueError("y_score must contain only finite values")

    labels = np.unique(true)
    if labels.size != 2 or not np.any(labels == pos_label):
        raise ValueError("ROC requires two classes and a present pos_label")

    positive = true == pos_label
    order = np.argsort(score, kind="stable")[::-1]
    sorted_score = score[order]
    sorted_positive = positive[order]
    threshold_indices = np.r_[
        np.flatnonzero(np.diff(sorted_score) != 0),
        sorted_score.size - 1,
    ]
    true_positives = np.cumsum(sorted_positive)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    thresholds = np.r_[np.inf, sorted_score[threshold_indices]]
    positives = int(np.sum(positive))
    negatives = positive.size - positives
    return (
        false_positives.astype(float) / negatives,
        true_positives.astype(float) / positives,
        thresholds,
    )


def roc_auc_score(y_true: Any, y_score: Any, *, pos_label: Any = 1) -> float:
    """Return the area under the binary receiver-operating curve."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, y_score, pos_label=pos_label
    )
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(true_positive_rate, false_positive_rate))


# Concise aliases required by the assignment plan.
accuracy = accuracy_score
precision = precision_score
recall = recall_score
f1 = f1_score
mse = mean_squared_error


__all__ = [
    "accuracy",
    "accuracy_score",
    "confusion_matrix",
    "f1",
    "f1_score",
    "mean_squared_error",
    "mse",
    "precision",
    "precision_score",
    "recall",
    "recall_score",
    "roc_auc_score",
    "roc_curve",
    "StreamingMetric",
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingAUC",
]


# ==========================================================================
# Streaming metrics
# ==========================================================================
from collections import deque  # noqa: E402


class StreamingMetric:
    """Base class: accumulate a metric over chunks, globally or in a window.

    Subclasses define how one chunk is summarised (a NumPy array of counts)
    and how a summary becomes a number. Because the summaries are additive,
    the global result is exact and a rolling window only needs the last
    ``window`` chunk summaries.

    Parameters
    ----------
    window : int or None
        None = everything since the last ``reset()``. An int W = only the
        last W ``update`` calls (rolling-window metric).

    Attributes
    ----------
    history_ : list
        ``result()`` after every update, ready to plot over time.
    n_updates_ : int
    """

    def __init__(self, window=None):
        if window is not None and (not isinstance(window, (int, np.integer))
                                   or isinstance(window, bool) or window < 1):
            raise ValueError(f"window must be None or an int >= 1, got {window!r}")
        self.window = window
        self.reset()

    def reset(self):
        """Forget all accumulated chunks and the history. Returns self."""
        self._total = None
        self._recent = deque()
        self.history_ = []
        self.n_updates_ = 0
        return self

    def _add(self, summary):
        """Add one chunk summary to the total (and drop the oldest in window mode)."""
        if self._total is None:
            self._total = summary.copy()
        else:
            self._total = self._total + summary
        if self.window is not None:
            self._recent.append(summary)
            if len(self._recent) > self.window:
                self._total = self._total - self._recent.popleft()
        self.n_updates_ += 1
        self.history_.append(self.result())

    def __repr__(self):
        return f"{type(self).__name__}(window={self.window})"


class StreamingConfusionMatrix(StreamingMetric):
    """Confusion matrix that accumulates over chunks (rows = true, cols = predicted).

    Parameters
    ----------
    labels : sequence or None
        Fixed label order. If None, labels are discovered as they arrive and
        kept sorted; the matrix grows (existing counts are moved, not lost).
    window : int or None
        Rolling window size in chunks.
    """

    def __init__(self, labels=None, window=None):
        self.fixed_labels = None if labels is None else np.asarray(list(labels))
        super().__init__(window)

    def reset(self):
        super().reset()
        self.labels_ = self.fixed_labels
        return self

    def _grow(self, new_labels):
        """Re-index every stored matrix when new labels appear."""
        old = self.labels_
        self.labels_ = new_labels
        if old is None or self._total is None:
            return
        pos = np.searchsorted(new_labels, old)

        def expand(m):
            out = np.zeros((len(new_labels), len(new_labels)), dtype=np.int64)
            out[np.ix_(pos, pos)] = m
            return out
        self._total = expand(self._total)
        self._recent = deque(expand(m) for m in self._recent)

    def _chunk_matrix(self, y_true, y_pred):
        true = np.asarray(y_true)
        pred = np.asarray(y_pred)
        if true.ndim != 1 or pred.ndim != 1:
            raise ValueError(f"y_true and y_pred must be 1-D, got shapes {true.shape} "
                             f"and {pred.shape}")
        if true.shape != pred.shape:
            raise ValueError(f"y_true and y_pred must have the same length, got "
                             f"{true.shape[0]} and {pred.shape[0]}")
        seen = np.unique(np.concatenate([true, pred])) if true.size else np.empty(0, true.dtype)
        if self.fixed_labels is not None:
            unknown = seen[~np.isin(seen, self.fixed_labels)]
            if unknown.size:
                raise ValueError(f"labels {list(unknown)} are not in the fixed labels")
            if self.labels_ is None:
                self.labels_ = self.fixed_labels
        elif self.labels_ is None or not np.isin(seen, self.labels_).all():
            self._grow(seen if self.labels_ is None else np.union1d(self.labels_, seen))
        k = len(self.labels_)
        if self.fixed_labels is not None:
            # fixed labels may be unsorted: map through a sorted view
            order = np.argsort(self.labels_)
            ti = order[np.searchsorted(self.labels_[order], true)]
            pi = order[np.searchsorted(self.labels_[order], pred)]
        else:
            ti, pi = np.searchsorted(self.labels_, true), np.searchsorted(self.labels_, pred)
        # one bincount instead of a k x k Python loop
        return np.bincount(ti * k + pi, minlength=k * k).reshape(k, k).astype(np.int64)

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of true and predicted labels. Returns self."""
        m = self._chunk_matrix(y_true_chunk, y_pred_chunk)   # grows stored matrices if needed
        self._add(m)
        return self

    def result(self):
        """Current (k, k) int matrix; empty (0, 0) before the first update."""
        if self._total is None:
            return np.zeros((0, 0), dtype=np.int64)
        return self._total.copy()


class StreamingAccuracy(StreamingConfusionMatrix):
    """Fraction of correct predictions over the stream (or the window).

    ``result()`` is NaN before any sample has been seen.
    """

    def __init__(self, window=None):
        super().__init__(labels=None, window=window)

    def result(self):
        m = super().result()
        total = m.sum()
        return float(np.trace(m) / total) if total else float("nan")


class _StreamingPRF(StreamingConfusionMatrix):
    """Precision / recall / F1 from the accumulated confusion matrix."""

    _metric = None

    def __init__(self, average="macro", pos_label=1, zero_division=0, window=None):
        if average not in ("binary", "micro", "macro", "weighted", None):
            raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted' or None")
        if zero_division not in (0, 1):
            raise ValueError("zero_division must be 0 or 1")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        super().__init__(labels=None, window=window)

    def result(self):
        m = super().result()
        if m.sum() == 0:
            return float("nan") if self.average is not None else np.zeros(0)
        tp = np.diag(m).astype(float)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp
        support = m.sum(axis=1).astype(float)
        zd = float(self.zero_division)
        if self.average == "micro":
            tp, fp, fn = tp.sum(keepdims=True), fp.sum(keepdims=True), fn.sum(keepdims=True)
        p = _safe_divide(tp, tp + fp, self.zero_division)
        r = _safe_divide(tp, tp + fn, self.zero_division)
        values = {"precision": p, "recall": r,
                  "f1": _safe_divide(2 * p * r, p + r, self.zero_division)}[self._metric]
        if self.average is None:
            return values
        if self.average == "micro":
            return float(values[0])
        if self.average == "binary":
            hit = np.flatnonzero(self.labels_ == self.pos_label)
            if len(self.labels_) > 2:
                raise ValueError("average='binary' needs at most two labels; "
                                 f"seen {list(self.labels_)}")
            return float(values[hit[0]]) if hit.size else zd
        if self.average == "weighted":
            return float(np.average(values, weights=support)) if support.sum() else zd
        return float(values.mean())


class StreamingPrecision(_StreamingPRF):
    """Streaming precision, TP / (TP + FP). See :class:`_StreamingPRF` options."""
    _metric = "precision"


class StreamingRecall(_StreamingPRF):
    """Streaming recall, TP / (TP + FN)."""
    _metric = "recall"


class StreamingF1(_StreamingPRF):
    """Streaming F1, the harmonic mean of precision and recall."""
    _metric = "f1"


class StreamingAUC(StreamingMetric):
    """Area under the ROC curve, accumulated over chunks in fixed memory.

    Scores are counted into ``n_bins`` equal bins on [0, 1], separately for
    positives and negatives. The AUC is the probability that a random positive
    scores higher than a random negative (ties, i.e. the same bin, count 1/2):

        AUC = sum_b pos_b * (neg_below_b + neg_b / 2) / (P * N)

    This is exact when all scores fall on distinct bins (a tree's scores are
    leaf fractions, so with the default 1000 bins it matches the batch
    ``roc_auc_score`` to about 1e-3 or better) and uses O(n_bins) memory no
    matter how long the stream is.

    Binary: ``update(y_true, y_score)`` with a 1-D score for ``pos_label``.
    Multiclass: pass ``classes`` and a 2-D ``y_score`` of shape
    (n_samples, n_classes) (e.g. ``predict_proba``); the result is the macro
    average of one-vs-rest AUCs over classes that have both positives and
    negatives so far.

    ``result()`` is NaN until both a positive and a negative have been seen.
    """

    def __init__(self, pos_label=1, classes=None, n_bins=1000, window=None):
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
            raise ValueError("n_bins must be an int >= 2")
        self.pos_label = pos_label
        self.classes = None if classes is None else np.asarray(list(classes))
        self.n_bins = int(n_bins)
        super().__init__(window)

    def update(self, y_true_chunk, y_score_chunk):
        true = np.asarray(y_true_chunk)
        score = np.asarray(y_score_chunk, dtype=float)
        if true.ndim != 1:
            raise ValueError(f"y_true must be 1-D, got shape {true.shape}")
        if score.shape[0] != true.shape[0]:
            raise ValueError(f"y_true has {true.shape[0]} samples but y_score has "
                             f"{score.shape[0]}")
        if np.isnan(score).any() or (score.size and (score.min() < 0 or score.max() > 1)):
            raise ValueError("y_score must be probabilities in [0, 1] without NaN")
        if score.ndim == 1:
            pos = (true == self.pos_label)[None, :]
            score = score[None, :]
        elif score.ndim == 2:
            if self.classes is None or len(self.classes) != score.shape[1]:
                raise ValueError("a 2-D y_score needs classes=[...] with one entry per column")
            pos = (true[None, :] == self.classes[:, None])
            score = score.T
        else:
            raise ValueError(f"y_score must be 1-D or 2-D, got shape {score.shape}")
        c = score.shape[0]
        b = np.minimum((score * self.n_bins).astype(np.intp), self.n_bins - 1)
        # index = class * (2 * n_bins) + is_positive * n_bins + bin
        idx = (np.arange(c)[:, None] * 2 * self.n_bins + pos * self.n_bins + b).ravel()
        counts = np.bincount(idx, minlength=c * 2 * self.n_bins).reshape(c, 2, self.n_bins)
        self._add(counts.astype(np.int64))
        return self

    def result(self):
        if self._total is None:
            return float("nan")
        neg, pos = self._total[:, 0, :].astype(float), self._total[:, 1, :].astype(float)
        P, N = pos.sum(axis=1), neg.sum(axis=1)
        neg_below = np.cumsum(neg, axis=1) - neg
        wins = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
        ok = (P > 0) & (N > 0)
        if not ok.any():
            return float("nan")
        return float(np.mean(wins[ok] / (P[ok] * N[ok])))
