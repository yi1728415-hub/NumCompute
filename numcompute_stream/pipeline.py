"""Composable transformer and estimator pipelines, with streaming support.

``Pipeline.partial_fit(X_chunk, y_chunk)`` passes one chunk through the
chain: every transformer first folds the chunk into its running state
(``partial_fit``), then transforms it, and the transformed chunk goes to the
next step; the final estimator receives it through its own ``partial_fit``
(for ``DecisionTreeClassifier`` that means: add to the buffer, retrain when
full). Prediction always uses each step's current state.

Note the order: a scaler is updated *before* it transforms the chunk, so the
model always sees data scaled with all statistics known at that moment. Rows
already sitting in the tree's buffer were scaled with the older statistics;
with a stable stream the difference is small, and the report discusses it.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .exceptions import NotFittedError


def _call_with_optional_y(method: Any, X: Any, y: Any, params: dict[str, Any]) -> Any:
    """Pass y only when the bound method's signature accepts it."""

    if y is None:
        return method(X, **params)
    try:
        inspect.signature(method).bind(X, y, **params)
    except (TypeError, ValueError):
        return method(X, **params)
    return method(X, y, **params)


class Pipeline:
    """Chain transformers and finish with a transformer or estimator."""

    def __init__(self, steps: Sequence[tuple[str, Any]]):
        self.steps = list(steps)
        self._validate_steps()
        self._is_fitted = False

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("steps must contain at least one (name, component) pair")
        names: list[str] = []
        for item in self.steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each pipeline step must be a (name, component) tuple")
            name, component = item
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("step names must be non-empty strings without '__'")
            if name in names:
                raise ValueError(f"duplicate step name: {name!r}")
            if component is None:
                raise ValueError(f"step {name!r} must not be None")
            names.append(name)

        for name, transformer in self.steps[:-1]:
            if not callable(getattr(transformer, "fit", None)):
                raise TypeError(f"intermediate step {name!r} must implement fit")
            if not callable(getattr(transformer, "transform", None)):
                raise TypeError(f"intermediate step {name!r} must implement transform")
        last_name, final = self.steps[-1]
        if not callable(getattr(final, "fit", None)):
            raise TypeError(f"final step {last_name!r} must implement fit")

    @property
    def named_steps(self) -> OrderedDict[str, Any]:
        """Return a name-to-step mapping in pipeline order."""
        return OrderedDict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int | slice | str) -> Any:
        if isinstance(index, str):
            try:
                return self.named_steps[index]
            except KeyError as error:
                raise KeyError(f"unknown pipeline step {index!r}") from error
        if isinstance(index, slice):
            return Pipeline(self.steps[index])
        return self.steps[index][1]

    def _split_fit_params(self, fit_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
        routed = {name: {} for name, _ in self.steps}
        for key, value in fit_params.items():
            if "__" not in key:
                raise ValueError(
                    f"fit parameter {key!r} must use the 'step__parameter' form"
                )
            name, parameter = key.split("__", 1)
            if name not in routed:
                raise ValueError(f"fit parameter targets unknown step {name!r}")
            if not parameter:
                raise ValueError(f"fit parameter {key!r} has an empty parameter name")
            routed[name][parameter] = value
        return routed

    def _fit_transformers(
        self,
        X: Any,
        y: Any,
        routed_params: dict[str, dict[str, Any]],
    ) -> Any:
        transformed = X
        for name, transformer in self.steps[:-1]:
            params = routed_params[name]
            fit_transform = getattr(transformer, "fit_transform", None)
            if callable(fit_transform):
                transformed = _call_with_optional_y(fit_transform, transformed, y, params)
            else:
                _call_with_optional_y(transformer.fit, transformed, y, params)
                transformed = transformer.transform(transformed)
        return transformed

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "Pipeline":
        """Fit intermediate transformers, then fit the final step. Return self."""
        routed = self._split_fit_params(fit_params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        _call_with_optional_y(final.fit, transformed, y, routed[final_name])
        self._is_fitted = True
        return self

    def partial_fit(self, X: Any, y: Any = None, **fit_params: Any) -> "Pipeline":
        """Update every step with one chunk (streaming). Return self.

        Each intermediate step must implement ``partial_fit`` and
        ``transform``; the final step must implement ``partial_fit``.
        """
        routed = self._split_fit_params(fit_params)
        transformed = X
        for name, transformer in self.steps[:-1]:
            method = getattr(transformer, "partial_fit", None)
            if not callable(method):
                raise TypeError(f"step {name!r} ({type(transformer).__name__}) does not "
                                "implement partial_fit, so it cannot be used on a stream")
            _call_with_optional_y(method, transformed, y, routed[name])
            transformed = transformer.transform(transformed)
        final_name, final = self.steps[-1]
        method = getattr(final, "partial_fit", None)
        if not callable(method):
            raise TypeError(f"final step {final_name!r} ({type(final).__name__}) does not "
                            "implement partial_fit")
        _call_with_optional_y(method, transformed, y, routed[final_name])
        self._is_fitted = True
        return self

    def flush(self) -> "Pipeline":
        """Ask the final step to train on whatever it has buffered (if it can)."""
        method = getattr(self.steps[-1][1], "flush", None)
        if callable(method):
            method()
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Pipeline is not fitted; call fit() or partial_fit() "
                                 "before inference")

    def _transform_before_final(self, X: Any) -> Any:
        self._require_fitted()
        transformed = X
        for _, transformer in self.steps[:-1]:
            transformed = transformer.transform(transformed)
        return transformed

    def predict(self, X: Any) -> Any:
        """Transform X through fitted intermediate steps and predict with the final estimator."""
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "predict", None)):
            raise AttributeError(f"final step {final_name!r} does not implement predict")
        return final.predict(transformed)

    def predict_proba(self, X: Any) -> Any:
        """Transform X and return probabilities from the fitted final estimator."""
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "predict_proba", None)):
            raise AttributeError(f"final step {final_name!r} does not implement predict_proba")
        return final.predict_proba(transformed)

    def transform(self, X: Any) -> Any:
        """Apply each fitted transform in order and return the final array."""
        self._require_fitted()
        transformed = X
        for name, component in self.steps:
            if not callable(getattr(component, "transform", None)):
                raise AttributeError(f"step {name!r} does not implement transform")
            transformed = component.transform(transformed)
        return transformed

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit the pipeline on X and return the transformed output."""
        routed = self._split_fit_params(fit_params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        params = routed[final_name]
        fit_transform = getattr(final, "fit_transform", None)
        if callable(fit_transform):
            result = _call_with_optional_y(fit_transform, transformed, y, params)
        else:
            _call_with_optional_y(final.fit, transformed, y, params)
            if not callable(getattr(final, "transform", None)):
                raise AttributeError(
                    f"final step {final_name!r} does not implement transform"
                )
            result = final.transform(transformed)
        self._is_fitted = True
        return result

    def score(self, X: Any, y: Any, **params: Any) -> float:
        """Transform X and delegate scoring to the fitted final estimator."""
        transformed = self._transform_before_final(X)
        final_name, final = self.steps[-1]
        if not callable(getattr(final, "score", None)):
            raise AttributeError(f"final step {final_name!r} does not implement score")
        return float(final.score(transformed, y, **params))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return step parameters, including name__parameter entries when deep is true."""
        params: dict[str, Any] = {"steps": self.steps}
        for name, component in self.steps:
            params[name] = component
            if deep:
                getter = getattr(component, "get_params", None)
                nested = (
                    getter(deep=True)
                    if callable(getter)
                    else dict(getattr(component, "__dict__", {}))
                )
                params.update({f"{name}__{key}": value for key, value in nested.items()})
        return params

    def set_params(self, **params: Any) -> "Pipeline":
        """Set named steps or name__parameter values and return self."""
        if not params:
            return self
        if "steps" in params:
            self.steps = list(params.pop("steps"))
            self._validate_steps()
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        valid_names = {name for name, _ in self.steps}
        for key, value in params.items():
            if "__" in key:
                name, parameter = key.split("__", 1)
                if name not in valid_names:
                    raise ValueError(f"unknown pipeline step {name!r}")
                nested.setdefault(name, {})[parameter] = value
            elif key in valid_names:
                direct[key] = value
            else:
                raise ValueError(f"unknown pipeline parameter {key!r}")

        self.steps = [(name, direct.get(name, component)) for name, component in self.steps]
        for name, component in self.steps:
            if name not in nested:
                continue
            setter = getattr(component, "set_params", None)
            if callable(setter):
                setter(**nested[name])
            else:
                for parameter, value in nested[name].items():
                    if not hasattr(component, parameter):
                        raise ValueError(
                            f"step {name!r} has no parameter {parameter!r}"
                        )
                    setattr(component, parameter, value)
        self._validate_steps()
        self._is_fitted = False
        return self


class FeatureUnion:
    """Fit transformers in parallel and concatenate their feature outputs."""

    def __init__(self, transformer_list: Sequence[tuple[str, Any]]):
        self.transformer_list = list(transformer_list)
        self._validate_transformers()
        self._is_fitted = False

    def _validate_transformers(self) -> None:
        if not self.transformer_list:
            raise ValueError("transformer_list must contain at least one transformer")
        names: set[str] = set()
        for item in self.transformer_list:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each transformer must be a (name, transformer) tuple")
            name, transformer = item
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("transformer names must be non-empty strings without '__'")
            if name in names:
                raise ValueError(f"duplicate transformer name: {name!r}")
            if not callable(getattr(transformer, "fit", None)):
                raise TypeError(f"transformer {name!r} must implement fit")
            if not callable(getattr(transformer, "transform", None)):
                raise TypeError(f"transformer {name!r} must implement transform")
            names.add(name)

    @property
    def named_transformers(self) -> OrderedDict[str, Any]:
        """Return a mapping from branch names to transformer objects."""
        return OrderedDict(self.transformer_list)

    def _split_fit_params(self, fit_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
        routed = {name: {} for name, _ in self.transformer_list}
        for key, value in fit_params.items():
            if "__" not in key:
                raise ValueError(
                    f"fit parameter {key!r} must use the 'transformer__parameter' form"
                )
            name, parameter = key.split("__", 1)
            if name not in routed or not parameter:
                raise ValueError(f"invalid fit parameter target: {key!r}")
            routed[name][parameter] = value
        return routed

    @staticmethod
    def _as_feature_matrix(values: Any, expected_rows: int | None) -> np.ndarray:
        result = np.asarray(values)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        if result.ndim != 2:
            raise ValueError("transformer outputs must be one- or two-dimensional")
        if expected_rows is not None and result.shape[0] != expected_rows:
            raise ValueError("all transformers must return the same number of rows")
        return result

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "FeatureUnion":
        """Fit every branch on the same X and return self."""
        routed = self._split_fit_params(fit_params)
        for name, transformer in self.transformer_list:
            _call_with_optional_y(transformer.fit, X, y, routed[name])
        self._is_fitted = True
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "FeatureUnion":
        """Update every transformer with one chunk. Return self."""
        for name, transformer in self.transformer_list:
            method = getattr(transformer, "partial_fit", None)
            if not callable(method):
                raise TypeError(f"transformer {name!r} does not implement partial_fit")
            _call_with_optional_y(method, X, y, {})
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Concatenate fitted branch outputs along the feature axis, preserving sample rows."""
        if not self._is_fitted:
            raise RuntimeError("FeatureUnion is not fitted; call fit before transform")
        outputs: list[np.ndarray] = []
        expected_rows: int | None = None
        for _, transformer in self.transformer_list:
            output = self._as_feature_matrix(transformer.transform(X), expected_rows)
            expected_rows = output.shape[0]
            outputs.append(output)
        return np.concatenate(outputs, axis=1)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        """Fit every branch and concatenate its transformed features."""
        routed = self._split_fit_params(fit_params)
        outputs: list[np.ndarray] = []
        expected_rows: int | None = None
        for name, transformer in self.transformer_list:
            method = getattr(transformer, "fit_transform", None)
            if callable(method):
                values = _call_with_optional_y(method, X, y, routed[name])
            else:
                _call_with_optional_y(transformer.fit, X, y, routed[name])
                values = transformer.transform(X)
            output = self._as_feature_matrix(values, expected_rows)
            expected_rows = output.shape[0]
            outputs.append(output)
        self._is_fitted = True
        return np.concatenate(outputs, axis=1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return branch parameters, optionally including nested values."""
        params: dict[str, Any] = {"transformer_list": self.transformer_list}
        for name, transformer in self.transformer_list:
            params[name] = transformer
            if deep:
                getter = getattr(transformer, "get_params", None)
                nested = getter(deep=True) if callable(getter) else dict(
                    getattr(transformer, "__dict__", {})
                )
                params.update({f"{name}__{key}": value for key, value in nested.items()})
        return params


class ColumnTransformer:
    """Apply different transformers to different columns and stack the results.

    Parameters
    ----------
    transformers : list of (name, transformer, columns)
        ``columns`` is a list of integer column positions. Columns not listed
        are dropped. Each transformer needs ``partial_fit``/``fit`` and
        ``transform``.

    The input may be an object array mixing numbers and text (e.g.
    ``np.column_stack([X_num.astype(object), X_cat])``); every transformer
    only sees its own columns. Outputs are concatenated left to right in the
    order given, so put a transformer whose width can grow (OneHotEncoder) last.
    """

    def __init__(self, transformers):
        self.transformers = list(transformers)
        if not self.transformers:
            raise ValueError("transformers must not be empty")
        names = set()
        for item in self.transformers:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("each entry must be a (name, transformer, columns) tuple")
            name, tr, cols = item
            if name in names:
                raise ValueError(f"duplicate transformer name {name!r}")
            names.add(name)
            if not callable(getattr(tr, "transform", None)):
                raise TypeError(f"transformer {name!r} must implement transform")
            if not all(isinstance(c, (int, np.integer)) for c in cols):
                raise TypeError(f"columns of {name!r} must be integer positions")

    def _select(self, X, cols):
        arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        if cols and max(cols) >= arr.shape[1]:
            raise ValueError(f"column {max(cols)} requested but X has {arr.shape[1]} columns")
        return arr[:, list(cols)]

    def partial_fit(self, X, y=None):
        for name, tr, cols in self.transformers:
            method = getattr(tr, "partial_fit", None)
            if not callable(method):
                raise TypeError(f"transformer {name!r} does not implement partial_fit")
            method(self._select(X, cols))
        return self

    def fit(self, X, y=None):
        for _, tr, cols in self.transformers:
            tr.fit(self._select(X, cols))
        return self

    def transform(self, X):
        parts = [np.asarray(tr.transform(self._select(X, cols)), dtype=np.float64)
                 for _, tr, cols in self.transformers]
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def _named_components(components: Sequence[Any]) -> list[tuple[str, Any]]:
    counts: dict[str, int] = {}
    named: list[tuple[str, Any]] = []
    for component in components:
        base = component.__class__.__name__.lower()
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}{counts[base]}"
        named.append((name, component))
    return named


def make_pipeline(*components: Any) -> Pipeline:
    """Construct a pipeline with deterministic component names."""

    if not components:
        raise ValueError("make_pipeline requires at least one component")
    return Pipeline(_named_components(components))


def make_union(*transformers: Any) -> FeatureUnion:
    """Construct a feature union with deterministic transformer names."""

    if not transformers:
        raise ValueError("make_union requires at least one transformer")
    return FeatureUnion(_named_components(transformers))


__all__ = ["ColumnTransformer", "FeatureUnion", "Pipeline", "make_pipeline", "make_union"]
