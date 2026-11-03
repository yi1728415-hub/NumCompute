"""Streaming metrics, pipeline, StreamTrainer, io and visualise."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import numcompute_stream as ncs  # noqa: E402
from numcompute_stream import metrics as M  # noqa: E402
from numcompute_stream import visualise as vis  # noqa: E402


def labels(n=900, seed=0):
    rng = np.random.default_rng(seed)
    yt = rng.integers(0, 3, n)
    yp = np.where(rng.random(n) < 0.7, yt, rng.integers(0, 3, n))
    return yt, yp


def feed(metric, yt, yp, k=9):
    for a, b in zip(np.array_split(yt, k), np.array_split(yp, k)):
        metric.update(a, b)
    return metric


# ---------------------------------------------------------------- metrics
def test_streaming_accuracy_and_confusion_equal_batch():
    yt, yp = labels()
    assert feed(M.StreamingAccuracy(), yt, yp).result() == pytest.approx(M.accuracy_score(yt, yp))
    assert np.array_equal(feed(M.StreamingConfusionMatrix(), yt, yp).result(),
                          M.confusion_matrix(yt, yp))


@pytest.mark.parametrize("cls, fn", [(M.StreamingPrecision, M.precision_score),
                                     (M.StreamingRecall, M.recall_score),
                                     (M.StreamingF1, M.f1_score)])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_streaming_prf_equal_batch(cls, fn, average):
    yt, yp = labels()
    assert feed(cls(average=average), yt, yp).result() == pytest.approx(fn(yt, yp, average=average))


def test_binary_average_and_zero_division():
    m = M.StreamingPrecision(average="binary", pos_label=1).update([0, 0], [0, 0])
    assert m.result() == 0.0


def test_rolling_window_metric():
    yt, yp = labels()
    w = feed(M.StreamingAccuracy(window=2), yt, yp)
    assert w.result() == pytest.approx(M.accuracy_score(yt[-200:], yp[-200:]))
    assert len(w.history_) == 9


def test_confusion_matrix_grows_with_new_labels_and_resets():
    cm = M.StreamingConfusionMatrix().update([0, 0], [0, 0]).update([2, 1], [2, 0])
    assert list(cm.labels_) == [0, 1, 2]
    assert cm.result().tolist() == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert cm.reset().result().shape == (0, 0)
    assert np.isnan(M.StreamingAccuracy().result())


def test_streaming_auc_matches_batch_and_supports_multiclass():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    s = np.clip(0.3 * y + 0.7 * rng.random(2000), 0, 1).round(3)
    auc = M.StreamingAUC()
    for a, b in zip(np.array_split(y, 7), np.array_split(s, 7)):
        auc.update(a, b)
    assert auc.result() == pytest.approx(M.roc_auc_score(y, s), abs=1e-3)
    yt, _ = labels()
    perfect = np.eye(3)[yt]
    assert M.StreamingAUC(classes=[0, 1, 2]).update(yt, perfect).result() == pytest.approx(1.0)


@pytest.mark.parametrize("call, match", [
    (lambda: M.StreamingAccuracy().update([0, 1], [0]), "same length"),
    (lambda: M.StreamingAccuracy(window=0), "window"),
    (lambda: M.StreamingAUC().update([0, 1], [0.2, 1.5]), r"\[0, 1\]"),
    (lambda: M.StreamingAUC().update([0, 1], np.ones((2, 2))), "classes"),
    (lambda: M.StreamingF1(average="sum"), "average"),
])
def test_metric_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# ---------------------------------------------------------------- pipeline + trainer
def stream_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y


def make_pipe(bs=400):
    return ncs.Pipeline([("imp", ncs.Imputer()), ("sc", ncs.StandardScaler()),
                         ("tree", ncs.DecisionTreeClassifier(max_depth=5, buffer_size=bs))])


def test_pipeline_partial_fit_updates_every_step():
    X, y = stream_data()
    pipe = make_pipe()
    for Xc, yc in ncs.iter_chunks(X, y, 200):
        pipe.partial_fit(Xc, yc)
    # the imputer runs first, so the scaler sees every row without NaN
    assert np.all(pipe["sc"].n_samples_seen_ == len(X))
    assert np.allclose(pipe["imp"].statistics_, np.nanmean(X, axis=0))
    assert pipe["tree"].n_retrains_ == 5
    assert pipe.score(X, y) > 0.85


def test_pipeline_rejects_step_without_partial_fit():
    class FitOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X
    pipe = ncs.Pipeline([("f", FitOnly()), ("tree", ncs.DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="partial_fit"):
        pipe.partial_fit(np.zeros((2, 2)), [0, 1])


def test_pipeline_predict_before_training():
    with pytest.raises(ncs.NotFittedError):
        make_pipe().predict(np.zeros((1, 4)))


def test_column_transformer_with_growing_one_hot():
    X = np.array([[1.0, "a"], [2.0, "b"], [3.0, "a"], [4.0, "c"]], dtype=object)
    ct = ncs.ColumnTransformer([("num", ncs.StandardScaler(), [0]),
                                ("cat", ncs.OneHotEncoder(), [1])])
    ct.partial_fit(X[:3])
    assert ct.transform(X[:3]).shape == (3, 3)
    ct.partial_fit(X[3:])
    assert ct.transform(X).shape == (4, 4)


def test_stream_trainer_prequential_log():
    X, y = stream_data()
    tr = ncs.StreamTrainer(make_pipe(), metrics={"acc": ncs.StreamingAccuracy()})
    tr.run(ncs.iter_chunks(X, y, 200))
    scored = tr.get_log("scored")
    assert not scored[:2].any() and scored[2:].all()       # nothing to score before retrain 1
    assert tr.get_log("retrained").sum() == 5
    assert tr.summary()["cumulative_accuracy"] == pytest.approx(tr.metrics["acc"].result())
    assert (tr.get_log("memory_bytes") > 0).all()
    assert "chunk" in tr.format_log()


def test_stream_trainer_validates_model_and_metrics():
    with pytest.raises(TypeError, match="partial_fit"):
        ncs.StreamTrainer(object())
    with pytest.raises(TypeError, match="StreamingMetric"):
        ncs.StreamTrainer(make_pipe(), metrics={"bad": 1})


def test_memory_footprint_counts_buffer():
    t = ncs.DecisionTreeClassifier(buffer_size=10 ** 6)
    small = ncs.memory_footprint(t)
    t.partial_fit(np.zeros((1000, 10)), np.zeros(1000))
    assert ncs.memory_footprint(t) - small >= 1000 * 10 * 8


# ---------------------------------------------------------------- io
def test_load_and_stream_dataset(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b,colour,label\n1,2,red,0\n3,,blue,1\n5,6,red,1\n")
    d = ncs.load_dataset(f, "label")
    assert d.num_names == ["a", "b"] and d.cat_names == ["colour"]
    assert np.isnan(d.X_num[1, 1]) and list(d.y) == [0, 1, 1]
    chunks = list(ncs.stream_dataset(f, "label", chunk_size=2))
    assert [c[0].shape[0] for c in chunks] == [2, 1]
    with pytest.raises(ValueError, match="not found"):
        ncs.load_dataset(f, "nope")


def test_ragged_csv_and_chunk_size_errors(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="line 3"):
        ncs.load_dataset(f, "b")
    with pytest.raises(ValueError, match="chunk_size"):
        list(ncs.iter_chunks(np.zeros((3, 1)), np.zeros(3), 0))


# ---------------------------------------------------------------- visualise
def test_required_plots_return_figures_and_save(tmp_path):
    f1 = vis.plot_metric_over_time([0.5, np.nan, 0.7, 0.8], "acc", "Accuracy",
                                   retrain_points=[2], save_path=tmp_path / "a.png")
    f2 = vis.compare_models([0.5, 0.6], [0.4, 0.7], labels=("x", "y"))
    f3 = vis.plot_predictions_vs_ground_truth([0, 1, 2, 1], [0, 1, 1, 1],
                                              save_path=tmp_path / "p.png")
    assert all(isinstance(f, matplotlib.figure.Figure) for f in (f1, f2, f3))
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "p.png").exists()


def test_plot_input_errors():
    with pytest.raises(ValueError, match="1-D"):
        vis.plot_metric_over_time(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="length"):
        vis.plot_predictions_vs_ground_truth([0, 1], [0])
