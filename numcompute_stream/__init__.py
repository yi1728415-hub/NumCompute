"""NumCompute-Stream: a NumPy-only, decision-tree-based streaming ML framework.

Built on the Assignment 2.1 NumCompute package. Every learning component
supports chunk-wise updates (``partial_fit`` / ``update`` / ``update_stats``),
and the model is a from-scratch CART tree that retrains on a buffer of the
most recent ``buffer_size`` samples.

Quick start::

    import numcompute_stream as ncs

    pipe = ncs.Pipeline([
        ("impute", ncs.Imputer()),
        ("scale", ncs.StandardScaler()),
        ("model", ncs.DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ])
    trainer = ncs.StreamTrainer(pipe)
    trainer.run(ncs.iter_chunks(X, y, chunk_size=100))
"""

from . import (
    benchmarking,
    io,
    metrics,
    optim,
    pipeline,
    preprocessing,
    rank,
    sort_search,
    stats,
    stream,
    tree,
    utils,
)
from .exceptions import NotFittedError
from .io import Dataset, iter_chunks, load_csv, load_dataset, stream_dataset
from .metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingMetric,
    StreamingPrecision,
    StreamingRecall,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .pipeline import ColumnTransformer, FeatureUnion, Pipeline, make_pipeline
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import RunningStats, StreamingHistogram, StreamingStats, merge_moments
from .stream import StreamTrainer, memory_footprint
from .tree import DecisionTreeClassifier, entropy, gini

__version__ = "0.2.0"

__all__ = [
    "ColumnTransformer", "Dataset", "DecisionTreeClassifier", "FeatureUnion", "Imputer", "MinMaxScaler",
    "NotFittedError", "OneHotEncoder", "Pipeline", "RunningStats", "SimpleImputer",
    "StandardScaler", "StreamTrainer", "StreamingAUC", "StreamingAccuracy",
    "StreamingConfusionMatrix", "StreamingF1", "StreamingHistogram", "StreamingMetric",
    "StreamingPrecision", "StreamingRecall", "StreamingStats", "accuracy_score",
    "benchmarking", "confusion_matrix", "entropy", "f1_score", "gini", "io", "iter_chunks",
    "load_csv", "load_dataset", "make_pipeline", "memory_footprint", "merge_moments",
    "metrics", "optim", "pipeline", "precision_score", "preprocessing", "rank",
    "recall_score", "roc_auc_score", "sort_search", "stats", "stream", "stream_dataset",
    "tree", "utils",
]
