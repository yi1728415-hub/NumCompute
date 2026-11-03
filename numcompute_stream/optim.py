"""Numerical differentiation utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

DifferenceMethod = Literal["central", "forward", "backward"]


def _validate_point(x: Any) -> np.ndarray:
    point = np.asarray(x, dtype=float)
    if point.size == 0:
        raise ValueError("x must not be empty")
    if not np.all(np.isfinite(point)):
        raise ValueError("x must contain only finite values")
    return point.copy()


def _validate_options(epsilon: float, method: str) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    if method not in {"central", "forward", "backward"}:
        raise ValueError("method must be 'central', 'forward', or 'backward'")


def _evaluate(function: Callable[..., Any], x: np.ndarray, args: tuple[Any, ...], kwargs: dict[str, Any]) -> np.ndarray:
    value = np.asarray(function(x.copy(), *args, **kwargs), dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("function returned a non-finite value")
    return value


def _step(value: float, epsilon: float) -> float:
    return epsilon * max(1.0, abs(float(value)))


def jacobian(
    function: Callable[..., Any],
    x: Any,
    *args: Any,
    epsilon: float = 1e-6,
    method: DifferenceMethod = "central",
    **kwargs: Any,
) -> np.ndarray:
    """Approximate a function's Jacobian with finite differences.

    For ``f: R^n -> R^m`` the result has shape ``(m, n)``. More generally,
    its shape is ``f(x).shape + x.shape``. A scalar output therefore produces
    an array with the same shape as ``x``.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    _validate_options(epsilon, method)
    point = _validate_point(x)
    base = _evaluate(function, point, args, kwargs)
    result = np.empty(base.shape + point.shape, dtype=float)
    result_view = result.reshape(base.size, point.size)
    flat = point.reshape(-1)

    for index in range(flat.size):
        delta = _step(flat[index], epsilon)
        plus = point.copy().reshape(-1)
        minus = point.copy().reshape(-1)
        plus[index] += delta
        minus[index] -= delta

        if method == "central":
            difference = (
                _evaluate(function, plus.reshape(point.shape), args, kwargs)
                - _evaluate(function, minus.reshape(point.shape), args, kwargs)
            ) / (2.0 * delta)
        elif method == "forward":
            difference = (
                _evaluate(function, plus.reshape(point.shape), args, kwargs) - base
            ) / delta
        else:
            difference = (
                base - _evaluate(function, minus.reshape(point.shape), args, kwargs)
            ) / delta

        if difference.shape != base.shape:
            raise ValueError("function output shape changed while computing the Jacobian")
        result_view[:, index] = difference.reshape(-1)

    return result


def gradient(
    function: Callable[..., Any],
    x: Any,
    *args: Any,
    epsilon: float = 1e-6,
    method: DifferenceMethod = "central",
    **kwargs: Any,
) -> np.ndarray:
    """Approximate the gradient of a scalar-valued function."""

    point = _validate_point(x)
    probe = _evaluate(function, point, args, kwargs)
    if probe.ndim != 0:
        raise ValueError(
            "gradient requires a scalar-valued function; use jacobian for vector outputs"
        )
    return jacobian(
        function,
        point,
        *args,
        epsilon=epsilon,
        method=method,
        **kwargs,
    )


numerical_gradient = gradient
numerical_jacobian = jacobian


__all__ = ["gradient", "jacobian", "numerical_gradient", "numerical_jacobian"]
