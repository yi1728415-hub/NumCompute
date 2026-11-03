"""Benchmarks for Assignment 2.2 (NumCompute-Stream).

    python benchmark/bench_stream.py           # full run (about a minute)
    python benchmark/bench_stream.py --quick   # smaller sizes, for a smoke test

Part A  loop-based vs vectorised implementations, under streaming conditions:
        A1 tree split search (DecisionTreeClassifier.split_method)
        A2 tree prediction (per-sample walk vs all rows one level at a time)
        A3 running mean/variance (per-sample Welford vs per-chunk Chan merge)
        A4 confusion matrix (k x k loop from Assignment 2.1 vs one bincount)
Part B  effect of buffer_size on the demo stream: prequential accuracy, macro F1,
        number of retrains, retraining time and recovery after concept drift.

Writes benchmark/results_stream.md and three PNG figures next to it.
Timings use numcompute_stream.benchmarking (median of repeats after a warm-up).
"""

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numcompute_stream as ncs  # noqa: E402
from numcompute_stream import visualise as vis  # noqa: E402
from numcompute_stream.benchmarking import benchmark  # noqa: E402
from numcompute_stream.metrics import confusion_matrix as loop_confusion_matrix  # noqa: E402

OUT = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv
DATA = ROOT / "demo" / "data" / "room_occupancy_stream.csv"


def synthetic(n, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int) + (X[:, 3] > 1).astype(int)
    return X, y


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ======================================================================= A
def bench_split_search():
    sizes = [250, 500, 1000] if QUICK else [250, 500, 1000, 2000, 4000]
    rows, loop_t, vec_t = [], [], []
    for n in sizes:
        X, y = synthetic(n)
        res = {}
        for method in ("loop", "vectorised"):
            t = ncs.DecisionTreeClassifier(max_depth=6, split_method=method)
            res[method] = benchmark(t.fit, X, y, repeat=1 if method == "loop" else 5,
                                    warmup=0 if method == "loop" else 1).median
        same = np.array_equal(
            ncs.DecisionTreeClassifier(max_depth=6, split_method="loop").fit(X, y).predict(X),
            ncs.DecisionTreeClassifier(max_depth=6).fit(X, y).predict(X))
        loop_t.append(res["loop"])
        vec_t.append(res["vectorised"])
        rows.append([n, f"{res['loop'] * 1000:.1f}", f"{res['vectorised'] * 1000:.2f}",
                     f"{res['loop'] / res['vectorised']:.0f}x", "yes" if same else "NO"])
    return sizes, loop_t, vec_t, table(
        ["buffer rows retrained on", "loop (ms)", "vectorised (ms)", "speed-up", "same tree"], rows)


def loop_predict(tree, X):
    """Reference: walk the tree one sample at a time in Python."""
    out = np.empty(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        node = 0
        while tree._feature[node] >= 0:
            node = tree._left[node] if x[tree._feature[node]] <= tree._threshold[node] \
                else tree._right[node]
        out[i] = node
    return out


def bench_predict():
    X, y = synthetic(2000)
    tree = ncs.DecisionTreeClassifier(max_depth=8).fit(X, y)
    rows = []
    for n in ([100, 1000] if QUICK else [100, 1000, 10000]):
        Xc, _ = synthetic(n, seed=1)
        assert np.array_equal(loop_predict(tree, Xc), tree._apply(Xc))
        tl = benchmark(loop_predict, tree, Xc, repeat=3).median
        tv = benchmark(tree._apply, Xc, repeat=5).median
        rows.append([n, f"{tl * 1000:.2f}", f"{tv * 1000:.3f}", f"{tl / tv:.0f}x"])
    return table(["chunk rows", "loop (ms)", "vectorised (ms)", "speed-up"], rows)


def welford_loop(chunks):
    n, mean, m2 = 0, None, None
    for c in chunks:
        for x in c:                       # one sample at a time
            n += 1
            if mean is None:
                mean, m2 = x.copy(), np.zeros_like(x)
                continue
            delta = x - mean
            mean = mean + delta / n
            m2 = m2 + delta * (x - mean)
    return mean, m2 / n


def chan_chunks(chunks):
    s = ncs.StandardScaler()
    for c in chunks:
        s.partial_fit(c)
    return s.mean_, s.var_


def bench_running_stats():
    rows = []
    for n in ([2000, 20000] if QUICK else [2000, 20000, 100000]):
        X, _ = synthetic(n)
        chunks = np.array_split(X, max(1, n // 100))
        m1, v1 = welford_loop(chunks)
        m2, v2 = chan_chunks(chunks)
        assert np.allclose(m1, m2) and np.allclose(v1, v2)
        tl = benchmark(welford_loop, chunks, repeat=3).median
        tv = benchmark(chan_chunks, chunks, repeat=5).median
        rows.append([n, len(chunks), f"{tl * 1000:.1f}", f"{tv * 1000:.2f}", f"{tl / tv:.0f}x"])
    return table(["rows", "chunks", "per-sample loop (ms)", "per-chunk merge (ms)", "speed-up"], rows)


def bench_confusion():
    rows = []
    rng = np.random.default_rng(0)
    for k in (3, 10, 30):
        yt, yp = rng.integers(0, k, 5000), rng.integers(0, k, 5000)
        assert np.array_equal(loop_confusion_matrix(yt, yp),
                              ncs.StreamingConfusionMatrix().update(yt, yp).result())
        tl = benchmark(loop_confusion_matrix, yt, yp, repeat=5).median
        tv = benchmark(lambda: ncs.StreamingConfusionMatrix().update(yt, yp), repeat=5).median
        rows.append([k, f"{tl * 1000:.2f}", f"{tv * 1000:.3f}", f"{tl / tv:.0f}x"])
    return table(["classes", "k x k loop (ms)", "bincount (ms)", "speed-up"], rows)


# ======================================================================= B
def demo_stream(chunk_size=100):
    for X_num, X_cat, y in ncs.stream_dataset(DATA, "occupancy", chunk_size=chunk_size):
        yield np.column_stack([X_num.astype(object), X_cat]), y


def make_pipeline(buffer_size):
    prep = ncs.ColumnTransformer([
        ("num", ncs.Pipeline([("impute", ncs.Imputer()), ("scale", ncs.StandardScaler())]),
         list(range(6))),
        ("cat", ncs.OneHotEncoder(), [6]),
    ])
    return ncs.Pipeline([("prep", prep), ("tree", ncs.DecisionTreeClassifier(
        max_depth=6, buffer_size=buffer_size, accept_new_features=True))])


def bench_buffer_sizes():
    sizes = [100, 500, 2000] if QUICK else [100, 200, 500, 1000, 2000, 4000]
    rows, acc, total_t, curves = [], [], [], {}
    for bs in sizes:
        tr = ncs.StreamTrainer(make_pipeline(bs), metrics={
            "accuracy": ncs.StreamingAccuracy(), "f1": ncs.StreamingF1(),
            "rolling": ncs.StreamingAccuracy(window=5)})
        tr.run(demo_stream())
        tree = tr.model.named_steps["tree"]
        secs = np.array([r["seconds"] for r in tree.retrain_log_])
        chunk_acc = tr.get_log("chunk_accuracy")
        first = int(np.flatnonzero(tr.get_log("scored"))[0]) + 1
        before = np.nanmean(chunk_acc[40:50])
        after = np.nanmean(chunk_acc[50:60]) if first <= 50 else float("nan")
        s = tr.summary()
        acc.append(s["accuracy"])
        total_t.append(secs.sum())
        curves[bs] = tr.get_log("rolling")
        rows.append([bs, first, s["retrains"], f"{s['accuracy']:.3f}", f"{s['f1']:.3f}",
                     f"{before:.3f}", f"{after:.3f}", f"{secs.mean() * 1000:.1f}",
                     f"{secs.sum() * 1000:.0f}", f"{s['peak_memory_bytes'] / 1024:.0f}"])
    return sizes, acc, total_t, curves, table(
        ["buffer_size", "first scored chunk", "retrains", "prequential acc", "macro F1",
         "acc chunks 41-50", "acc chunks 51-60 (after drift)", "ms / retrain",
         "total retrain ms", "peak KiB"], rows)


def main():
    print("A1 split search ..."); sizes, lt, vt, t_a1 = bench_split_search()
    print("A2 prediction ..."); t_a2 = bench_predict()
    print("A3 running statistics ..."); t_a3 = bench_running_stats()
    print("A4 confusion matrix ..."); t_a4 = bench_confusion()
    print("B  buffer_size sweep ..."); bsz, acc, tt, curves, t_b = bench_buffer_sizes()

    fig = vis.compare_models(np.array(lt) * 1000, np.array(vt) * 1000,
                             labels=("loop", "vectorised"), xlabel="Rows in buffer",
                             ylabel="Retrain time (ms, log scale)",
                             title="Split search: loop vs vectorised")
    ax = fig.axes[0]
    for line in ax.get_lines():
        line.set_xdata(sizes)
    for txt in list(ax.texts):
        txt.remove()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.relim()
    ax.autoscale_view()
    fig.savefig(OUT / "split_search_loop_vs_vectorised.png", dpi=150, bbox_inches="tight")

    vis.plot_buffer_size_tradeoff(bsz, acc, tt, title="Effect of buffer_size on the demo stream",
                                  save_path=OUT / "buffer_size_tradeoff.png")
    # three representative sizes keep the plot readable
    fig = vis.compare_models(curves[100], curves[500], labels=("buffer_size=100", "buffer_size=500"),
                             extra={"buffer_size=2000": curves[2000]},
                             title="Rolling accuracy (last 5 chunks); drift at chunk 50",
                             ylabel="Accuracy")
    fig.axes[0].axvline(50, color="#52514e", alpha=0.5, linestyle=":")
    fig.axes[0].legend(frameon=False, fontsize=9, loc="lower left", ncol=3)
    fig.savefig(OUT / "buffer_size_rolling_accuracy.png", dpi=150, bbox_inches="tight")

    import matplotlib as mpl
    report = f"""# Benchmark results: NumCompute-Stream (Assignment 2.2)

Generated by `python benchmark/bench_stream.py`{' --quick' if QUICK else ''}.
Times are medians over repeats after a warm-up (the slow loop versions of A1 run once).
Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()},
NumPy {np.__version__}, matplotlib {mpl.__version__}. Absolute times vary by machine;
the ratios are what matter.

## A. Loop-based vs vectorised implementations

### A1. Tree split search (one retrain on the buffer, `max_depth=6`, 8 features)

The loop version tries every threshold of every feature in Python; the vectorised
version sorts each feature column once and gets the class counts left of every cut
from one `np.cumsum`, then scores all cuts at once. Both build the same tree.

{t_a1}

![split search](split_search_loop_vs_vectorised.png)

### A2. Prediction on an incoming chunk (tree of depth 8)

Loop: walk the tree sample by sample. Vectorised: move all rows down one level per
step with fancy indexing, so the Python loop runs `depth` times instead of `rows` times.

{t_a2}

### A3. Running mean / variance over 100-row chunks (8 features)

Loop: Welford's update one sample at a time. Vectorised: per-chunk moments with
NumPy, merged with Chan's formula (`StandardScaler.partial_fit`). Same results.

{t_a3}

### A4. Confusion matrix of a 5000-row chunk

Assignment 2.1 built it with a k x k loop of comparisons; the streaming version
encodes each (true, predicted) pair as one integer and calls `np.bincount` once.

{t_a4}

## B. Effect of `buffer_size` (demo stream: 8000 rows in 80 chunks of 100)

Pipeline: Imputer -> StandardScaler (numeric), OneHotEncoder (`day_type`), then
`DecisionTreeClassifier(max_depth=6)`. Prequential (test-then-train) evaluation.
The light sensor is replaced at row 5000 (chunk 50).

{t_b}

![buffer size trade-off](buffer_size_tradeoff.png)

![rolling accuracy](buffer_size_rolling_accuracy.png)

### Reading the results

* **Accuracy rises with buffer_size up to a point, then falls.** Each tree sees more
  rows, so its splits are less noisy. Very small buffers (100 rows is about 17 hours
  of readings) learn from too little data and do not cover a full day/week cycle.
  The largest buffer does worse again: its single tree is used for a long time and
  the stream has to wait longer before any tree exists at all.
* **A small buffer starts predicting sooner** (the first scored chunk is
  `buffer_size / 100 + 1`); with `buffer_size=4000` half the stream goes unscored.
* **Retraining time per retrain grows roughly linearly with buffer_size** (sorting
  dominates), but there are proportionally fewer retrains, so the *total* retraining
  time stays in the same range and even falls slightly.
* **Memory grows with buffer_size**: the peak footprint is the full buffer just
  before a retrain (about 8 bytes x 7 columns x buffer_size, plus Python overhead).
* **Drift**: a model only adapts after its next retrain, so the reaction delay is up
  to `buffer_size` rows. Every setting loses some accuracy in chunks 51-60; the
  drop is mild because CO2 and noise still carry the occupancy signal.
* A4 shows vectorising does not always pay at tiny sizes: with 3 classes the k x k
  loop is only 9 comparisons, so the two versions take about the same time.
"""
    (OUT / "results_stream.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
