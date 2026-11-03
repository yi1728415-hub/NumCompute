"""StreamTrainer: drive a model or pipeline over a stream and log everything.

Evaluation protocol: **prequential ("test-then-train")**. For each chunk the
current model first predicts it (``score_chunk``), then learns from it
(``fit_chunk``). Every chunk is therefore scored on data the model has not
trained on, which is the standard way to evaluate online learners and needs no
separate test set. Chunks that arrive before the first retrain cannot be
scored; they are logged with ``scored=False``.

Each call appends one record (a dict) to ``history_``:

=================  ======================================================
chunk              1-based chunk number
n_samples          rows in this chunk
n_seen             rows received so far (including this one)
scored             False while the model had not been trained yet
chunk_accuracy     accuracy on this chunk alone (NaN if not scored)
cumulative_accuracy  correct / scored over the whole stream so far
<metric name>      result() of every streaming metric passed in
retrained          True if this chunk triggered a retrain
n_retrains         retrains so far
n_buffered         rows waiting in the tree's buffer after this chunk
fit_seconds, predict_seconds   wall time of partial_fit / predict
memory_bytes       memory held by the model (arrays + buffer), see
                   :func:`memory_footprint`
=================  ======================================================
"""

import logging
import sys
import time
from collections import deque

import numpy as np

from ._validation import check_1d
from .exceptions import NotFittedError
from .metrics import StreamingAccuracy, StreamingF1, StreamingMetric

__all__ = ["StreamTrainer", "memory_footprint"]

logger = logging.getLogger("numcompute_stream")


def memory_footprint(obj, _seen=None):
    """Approximate bytes held by ``obj``: NumPy buffers plus container overhead.

    Walks attributes, lists, tuples, dicts and deques recursively and counts
    every array's ``nbytes`` once (shared arrays are not double counted).
    """
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + (memory_footprint(obj.base, seen) if obj.base is not None else 0)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(memory_footprint(k, seen) + memory_footprint(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum(memory_footprint(v, seen) for v in obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        size += memory_footprint(vars(obj), seen)
    return int(size)


def _final_estimator(model):
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


class StreamTrainer:
    """Train a model chunk by chunk, score it prequentially and keep a log.

    Parameters
    ----------
    model : estimator or Pipeline
        Must implement ``partial_fit(X, y)`` and ``predict(X)``.
        ``predict_proba`` is used by metrics that need scores (AUC).
    metrics : dict of {name: StreamingMetric} or None
        Updated with every scored chunk. Default: cumulative accuracy,
        macro F1 and a rolling accuracy over the last 5 chunks.
    verbose : bool, default False
        Log one line per chunk at INFO level (and print it).

    Attributes
    ----------
    history_ : list of dict
    last_y_true_, last_y_pred_, last_proba_ : arrays from the latest scored chunk
    """

    def __init__(self, model, metrics=None, verbose=False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model must implement {method}(); got {type(model).__name__}")
        if metrics is None:
            metrics = {"accuracy": StreamingAccuracy(),
                       "f1_macro": StreamingF1(average="macro"),
                       "rolling_accuracy": StreamingAccuracy(window=5)}
        for name, m in metrics.items():
            if not isinstance(m, StreamingMetric):
                raise TypeError(f"metric {name!r} must be a StreamingMetric, got {type(m).__name__}")
        self.model = model
        self.metrics = dict(metrics)
        self.verbose = verbose
        self.reset_log()

    def reset_log(self):
        """Clear the history and reset every metric (the model is kept)."""
        self.history_ = []
        self._n_seen = 0
        self._correct = 0
        self._scored = 0
        self._pending = None
        self.last_y_true_ = self.last_y_pred_ = self.last_proba_ = None
        for m in self.metrics.values():
            m.reset()
        return self

    # ------------------------------------------------------------- scoring
    def score_chunk(self, X, y):
        """Predict X with the current model, update metrics, return a dict.

        Does not train. If the model has not been trained yet the dict has
        ``scored=False`` and NaN accuracy.
        """
        y = check_1d(y, "y")
        rec = {"scored": False, "chunk_accuracy": float("nan"), "predict_seconds": 0.0}
        start = time.perf_counter()
        try:
            y_pred = np.asarray(self.model.predict(X))
        except NotFittedError:
            return rec
        proba = None
        needs_proba = any(type(m).__name__ == "StreamingAUC" for m in self.metrics.values())
        if needs_proba and callable(getattr(self.model, "predict_proba", None)):
            proba = self.model.predict_proba(X)
        rec["predict_seconds"] = time.perf_counter() - start
        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"X has {y_pred.shape[0]} rows but y has {y.shape[0]}")
        correct = int(np.sum(y_pred == y))
        self._correct += correct
        self._scored += y.shape[0]
        rec.update(scored=True, chunk_accuracy=correct / y.shape[0] if y.size else float("nan"))
        for m in self.metrics.values():
            if type(m).__name__ == "StreamingAUC":
                if proba is not None:
                    est = _final_estimator(self.model)
                    m.classes = np.asarray(est.classes_)
                    m.update(y, proba if proba.shape[1] > 2 else proba[:, -1])
            else:
                m.update(y, y_pred)
        self.last_y_true_, self.last_y_pred_, self.last_proba_ = y, y_pred, proba
        return rec

    # ------------------------------------------------------------ training
    def fit_chunk(self, X, y):
        """Feed one chunk to ``model.partial_fit``; return a dict of what happened."""
        est = _final_estimator(self.model)
        before = getattr(est, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        seconds = time.perf_counter() - start
        self._n_seen += int(np.asarray(y).shape[0])
        after = getattr(est, "n_retrains_", 0)
        return {"fit_seconds": seconds, "retrained": after > before, "n_retrains": after,
                "n_buffered": getattr(est, "n_buffered_", 0)}

    def process_chunk(self, X, y):
        """Test-then-train on one chunk and append the record to ``history_``."""
        y = check_1d(y, "y")
        rec = {"chunk": len(self.history_) + 1, "n_samples": int(y.shape[0])}
        rec.update(self.score_chunk(X, y))
        rec.update(self.fit_chunk(X, y))
        rec["n_seen"] = self._n_seen
        rec["cumulative_accuracy"] = self._correct / self._scored if self._scored else float("nan")
        for name, m in self.metrics.items():
            res = m.result()
            rec[name] = res if np.ndim(res) == 0 else np.asarray(res)
        rec["memory_bytes"] = memory_footprint(self.model)
        self.history_.append(rec)
        if self.verbose:
            line = (f"chunk {rec['chunk']:>3} | seen {rec['n_seen']:>6} | "
                    f"acc {rec['chunk_accuracy']:.3f} | cum {rec['cumulative_accuracy']:.3f} | "
                    f"retrains {rec['n_retrains']:>2}{' *' if rec['retrained'] else '  '} | "
                    f"buffer {rec['n_buffered']:>5} | mem {rec['memory_bytes'] / 1024:.1f} KiB")
            logger.info(line)
            print(line)
        return rec

    def run(self, stream, max_chunks=None, flush=False):
        """Process every ``(X, y)`` pair from an iterable. Returns ``history_``.

        ``flush=True`` trains on the leftover buffer after the last chunk.
        """
        for i, (X, y) in enumerate(stream):
            if max_chunks is not None and i >= max_chunks:
                break
            self.process_chunk(X, y)
        if flush and callable(getattr(self.model, "flush", None)):
            self.model.flush()
        return self.history_

    # ---------------------------------------------------------------- logs
    def get_log(self, key):
        """One column of the history as an array (e.g. ``get_log("accuracy")``)."""
        if self.history_ and key not in self.history_[0]:
            raise KeyError(f"unknown log key {key!r}; available: {list(self.history_[0])}")
        return np.array([r[key] for r in self.history_])

    def summary(self):
        """Dict with the final value of every logged quantity worth reporting."""
        if not self.history_:
            return {}
        last = self.history_[-1]
        out = {"chunks": len(self.history_), "samples": last["n_seen"],
               "retrains": last["n_retrains"],
               "cumulative_accuracy": last["cumulative_accuracy"],
               "total_fit_seconds": float(self.get_log("fit_seconds").sum()),
               "peak_memory_bytes": int(self.get_log("memory_bytes").max())}
        for name in self.metrics:
            out[name] = last[name]
        return out

    def format_log(self, keys=("chunk", "n_seen", "chunk_accuracy", "cumulative_accuracy",
                               "retrained", "n_buffered", "memory_bytes")):
        """The history as a plain-text table."""
        rows = [[str(k) for k in keys]]
        for r in self.history_:
            rows.append([f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys])
        widths = [max(len(row[i]) for row in rows) for i in range(len(keys))]
        return "\n".join("  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in rows)

    def to_csv(self, path, keys=None):
        """Write scalar columns of the history to a CSV file."""
        if not self.history_:
            raise ValueError("nothing logged yet")
        keys = keys or [k for k, v in self.history_[0].items() if np.ndim(v) == 0]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in self.history_:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        return path
