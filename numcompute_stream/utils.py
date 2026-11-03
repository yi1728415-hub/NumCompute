import numpy as np

def logsumexp(x, axis=None, keepdims=False):
    """Compute log(sum(exp(x))) along axis using a maximum shift.

    axis=None reduces all elements. keepdims retains reduced dimensions.
    Time is O(x.size); temporary arrays also use O(x.size) storage."""
    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)

    result = max_x + np.log(
        np.sum(
            np.exp(x - max_x),
            axis=axis,
            keepdims=True
        )
    )

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return result

def softmax(x, axis=-1):
    """Return probabilities normalised along axis, preserving x.shape.

    Subtracting the maximum avoids exponential overflow for large finite
    inputs. Time and temporary storage are O(x.size)."""
    x = np.asarray(x, dtype=float)

    shifted = x - np.max(
        x,
        axis=axis,
        keepdims=True
    )

    exp_x = np.exp(shifted)

    return exp_x / np.sum(
        exp_x,
        axis=axis,
        keepdims=True
    )


def sigmoid(x):
    """Numerically stable logistic activation."""

    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def relu(x):
    """Rectified linear activation."""

    return np.maximum(np.asarray(x), 0)

def euclidean_distance(a, b):
    """Return Euclidean distances along the last axis of broadcastable a and b.

    Leading dimensions are preserved. Incompatible shapes raise ValueError.
    Time and temporary storage scale with the broadcasted array size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sqrt(
        np.sum((a - b) ** 2, axis=-1)
    )

def batch_iterator(X, batch_size):
    """Yield slices of X with at most batch_size rows, preserving row order.

    A nonpositive batch size raises ValueError. Slicing an ndarray returns
    views, so iteration does not copy the full input."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size]


def topk_indices(values, k, largest=True):
    """Return sorted indices of the ``k`` largest or smallest 1-D values."""

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer")
    if k < 0 or k > values.size:
        raise ValueError("k must satisfy 0 <= k <= len(values)")
    if k == 0:
        return np.empty(0, dtype=np.intp)

    if largest:
        selected = np.argpartition(values, values.size - k)[-k:]
        order = np.argsort(values[selected], kind="stable")[::-1]
    else:
        selected = np.argpartition(values, k - 1)[:k]
        order = np.argsort(values[selected], kind="stable")
    return selected[order]


__all__ = [
    "batch_iterator",
    "euclidean_distance",
    "logsumexp",
    "relu",
    "sigmoid",
    "softmax",
    "topk_indices",
]
