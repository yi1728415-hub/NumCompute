"""Reusable matplotlib plots for streaming experiments.

Required plots
--------------
plot_metric_over_time(metric_values, title, ylabel)
compare_models(metric1, metric2, labels)
plot_predictions_vs_ground_truth(y_true, y_pred)

Extras used by the demo and benchmarks
--------------------------------------
plot_confusion_matrix, plot_retrain_profile, plot_buffer_size_tradeoff,
plot_stream_dashboard.

Every function

* takes an optional ``ax`` to draw into an existing figure (or makes one),
* takes ``save_path`` to write the figure to a file (PNG/PDF/SVG by extension),
* takes ``show`` to call ``plt.show()`` (inline display in a notebook),
* returns the ``matplotlib.figure.Figure``.

NaN values (e.g. chunks scored before the first retrain) are left as gaps.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - matplotlib is a declared dependency
    raise ImportError("visualise needs matplotlib: pip install matplotlib") from exc

__all__ = [
    "plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth",
    "plot_confusion_matrix", "plot_retrain_profile", "plot_buffer_size_tradeoff",
    "plot_stream_dashboard", "PALETTE",
]

# Colour-blind-checked categorical order (blue, orange, aqua, ...), fixed so a
# model keeps its colour across every figure. Ink colours are for text/axes.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948"]
_INK, _INK_SOFT, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
_SEQ = "Blues"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_SOFT)
    ax.tick_params(colors=_INK_SOFT, labelsize=9)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=_INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_SOFT, fontsize=10)


def _new_ax(ax, figsize=(8, 4)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, facecolor=_SURFACE)
    return fig, ax


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def _as_series(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one value per chunk), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", *,
                          x=None, xlabel="Chunk", label=None, retrain_points=None,
                          event_label="retrain", color=None, ax=None, save_path=None,
                          show=False):
    """Line plot of one streaming metric, one point per chunk.

    Parameters
    ----------
    metric_values : 1-D array-like
        e.g. ``trainer.get_log("cumulative_accuracy")``.
    x : 1-D array-like or None
        x positions (default 1..n chunks).
    retrain_points : 1-D array-like of x positions, or bool mask, or None
        Draw a faint vertical line where the tree was retrained (or at any
        other event, named by ``event_label``).
    """
    y = _as_series(metric_values, "metric_values")
    xs = np.arange(1, y.size + 1) if x is None else _as_series(x, "x")
    if xs.size != y.size:
        raise ValueError(f"x has {xs.size} values but metric_values has {y.size}")
    fig, ax = _new_ax(ax)
    if retrain_points is not None:
        rp = np.asarray(retrain_points)
        rp = xs[rp] if rp.dtype == bool else rp
        for i, p in enumerate(rp):
            ax.axvline(p, color=_INK_SOFT, alpha=0.25, linewidth=1, linestyle="--",
                       label=event_label if i == 0 else None)
    ax.plot(xs, y, color=color or PALETTE[0], linewidth=2, marker="o", markersize=4,
            label=label)
    last = np.flatnonzero(~np.isnan(y))
    if last.size:
        j = last[-1]
        ax.annotate(f"{y[j]:.3f}", (xs[j], y[j]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=_INK)
    _style(ax, title, xlabel, ylabel)
    if label or retrain_points is not None:
        ax.legend(frameon=False, fontsize=9)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model A", "Model B"), *,
                   title="Model comparison", ylabel="Metric", xlabel="Chunk",
                   extra=None, ax=None, save_path=None, show=False):
    """Overlay the same streaming metric for two (or more) models.

    Parameters
    ----------
    metric1, metric2 : 1-D array-like, one value per chunk
    labels : pair of str
    extra : dict {label: values} or None
        More models to add (e.g. a sweep over buffer_size); up to 8 in total.
    """
    series = {labels[0]: _as_series(metric1, "metric1"), labels[1]: _as_series(metric2, "metric2")}
    if extra:
        series.update({k: _as_series(v, k) for k, v in extra.items()})
    if len(series) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} models can be compared in one plot")
    fig, ax = _new_ax(ax)
    for i, (name, vals) in enumerate(series.items()):
        xs = np.arange(1, vals.size + 1)
        ax.plot(xs, vals, color=PALETTE[i], linewidth=2, label=name)
        ok = np.flatnonzero(~np.isnan(vals))
        if ok.size and len(series) <= 4:
            ax.annotate(f"{vals[ok[-1]]:.3f}", (xs[ok[-1]], vals[ok[-1]]),
                        textcoords="offset points", xytext=(6, 0), va="center",
                        fontsize=9, color=_INK)
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, save_path, show)


def plot_confusion_matrix(matrix, labels=None, *, title="Confusion matrix",
                          normalize=False, ax=None, save_path=None, show=False):
    """Heat map of a (k, k) confusion matrix (rows = true, columns = predicted)."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square, got shape {m.shape}")
    shown = m / np.maximum(m.sum(axis=1, keepdims=True), 1) if normalize else m
    fig, ax = _new_ax(ax, figsize=(4.5, 4))
    ax.imshow(shown, cmap=_SEQ, vmin=0)
    k = m.shape[0]
    names = [str(v) for v in (labels if labels is not None else range(k))]
    ax.set_xticks(range(k), names)
    ax.set_yticks(range(k), names)
    hi = shown.max() if shown.size else 1
    for i in range(k):
        for j in range(k):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(m[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > 0.6 * hi else _INK)
    ax.set_xlabel("Predicted", color=_INK_SOFT)
    ax.set_ylabel("True", color=_INK_SOFT)
    ax.set_title(title, loc="left", fontsize=12, color=_INK, fontweight="bold")
    ax.tick_params(colors=_INK_SOFT, labelsize=9)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title="Latest chunk: predictions vs ground truth",
                                     labels=None, max_points=200, save_path=None, show=False):
    """Two panels for the latest chunk: per-sample labels and the confusion matrix.

    Left: true label (hollow ring) and predicted label (dot) for each sample,
    with mistakes highlighted. Right: the chunk's confusion matrix.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.size} vs {yp.size}")
    if yt.size == 0:
        raise ValueError("y_true is empty")
    classes = np.asarray(labels) if labels is not None else np.unique(np.concatenate([yt, yp]))
    ti, pi = np.searchsorted(classes, yt), np.searchsorted(classes, yp)
    n = min(yt.size, max_points)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=_SURFACE,
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    idx = np.arange(n)
    wrong = ti[:n] != pi[:n]
    a1.scatter(idx, ti[:n], s=60, facecolors="none", edgecolors=_INK_SOFT, linewidths=1.2,
               label="true")
    a1.scatter(idx[~wrong], pi[:n][~wrong], s=18, color=PALETTE[0], label="predicted (correct)")
    a1.scatter(idx[wrong], pi[:n][wrong], s=26, color=PALETTE[1], marker="X",
               label="predicted (wrong)")
    a1.set_yticks(range(len(classes)), [str(c) for c in classes])
    acc = float(np.mean(ti == pi))
    _style(a1, f"{title} (accuracy {acc:.3f})", "Sample in chunk", "Class")
    a1.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    k = len(classes)
    cm = np.bincount(ti * k + pi, minlength=k * k).reshape(k, k)
    plot_confusion_matrix(cm, classes, title="Confusion matrix", ax=a2)
    return _finish(fig, save_path, show)


def plot_retrain_profile(retrain_log, *, title="Retraining cost", ax=None,
                         save_path=None, show=False):
    """Bar chart of seconds per retrain, from ``DecisionTreeClassifier.retrain_log_``."""
    if not retrain_log:
        raise ValueError("retrain_log is empty: the tree has not been trained yet")
    secs = np.array([r["seconds"] for r in retrain_log]) * 1000
    fig, ax = _new_ax(ax)
    ax.bar(np.arange(1, secs.size + 1), secs, color=PALETTE[0], width=0.6)
    _style(ax, title, "Retrain number", "Milliseconds")
    return _finish(fig, save_path, show)


def plot_buffer_size_tradeoff(buffer_sizes, accuracy, retrain_seconds, *,
                              title="Effect of buffer_size", save_path=None, show=False):
    """Two side-by-side panels (accuracy, total retrain time) against buffer_size.

    Kept as two panels with their own y-axes rather than one dual-axis chart.
    """
    b = _as_series(buffer_sizes, "buffer_sizes")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=_SURFACE)
    a1.plot(b, _as_series(accuracy, "accuracy"), color=PALETTE[0], marker="o", linewidth=2)
    _style(a1, "Prequential accuracy", "buffer_size (log scale)", "Accuracy")
    a2.plot(b, np.asarray(retrain_seconds) * 1000, color=PALETTE[1], marker="o", linewidth=2)
    _style(a2, "Total retraining time", "buffer_size (log scale)", "Milliseconds")
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(b, [str(int(v)) for v in b])
        a.minorticks_off()
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=_INK, fontweight="bold")
    return _finish(fig, save_path, show)


def plot_stream_dashboard(trainer, *, title="Streaming run", save_path=None, show=False):
    """Four panels from a :class:`~numcompute_stream.stream.StreamTrainer` log:
    accuracy (chunk, cumulative, rolling), buffer fill, fit time and memory."""
    h = trainer.history_
    if not h:
        raise ValueError("the trainer has no history yet")
    x = np.array([r["chunk"] for r in h])
    retr = np.array([r["retrained"] for r in h])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=_SURFACE)
    a = axes[0, 0]
    a.plot(x, [r["chunk_accuracy"] for r in h], color=PALETTE[0], alpha=0.45, linewidth=1.5,
           label="this chunk")
    a.plot(x, [r["cumulative_accuracy"] for r in h], color=PALETTE[0], linewidth=2.2,
           label="cumulative")
    if "rolling_accuracy" in h[0]:
        a.plot(x, [r["rolling_accuracy"] for r in h], color=PALETTE[1], linewidth=2,
               label="rolling (5 chunks)")
    for p in x[retr]:
        a.axvline(p, color=_INK_SOFT, alpha=0.2, linestyle="--", linewidth=1)
    _style(a, "Accuracy (dashed = retrain)", "Chunk", "Accuracy")
    a.legend(frameon=False, fontsize=8)
    a = axes[0, 1]
    a.bar(x, [r["n_buffered"] for r in h], color=PALETTE[2], width=0.7)
    _style(a, "Samples waiting in the buffer", "Chunk", "Rows")
    a = axes[1, 0]
    a.bar(x, np.array([r["fit_seconds"] for r in h]) * 1000,
          color=np.where(retr, PALETTE[1], PALETTE[0]), width=0.7)
    _style(a, "partial_fit time (orange = retrain)", "Chunk", "Milliseconds")
    a = axes[1, 1]
    a.plot(x, np.array([r["memory_bytes"] for r in h]) / 1024, color=PALETTE[0], linewidth=2,
           marker="o", markersize=3)
    _style(a, "Model memory footprint", "Chunk", "KiB")
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, color=_INK, fontweight="bold")
    return _finish(fig, save_path, show)
