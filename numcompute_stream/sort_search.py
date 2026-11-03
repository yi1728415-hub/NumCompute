"""Sorting, top-k selection and searching utilities.

All public functions accept array-like input and return NumPy arrays.
Core computations are vectorised; the only Python loops are over a small,
user-supplied number of sort keys (``multikey_sort``), over partition rounds
in ``quickselect`` (expected O(log n) rounds), and in ``binary_search_py``,
which is a deliberately loop-based reference version kept for benchmarking.

Functions
---------
stable_sort      Stable sort along an axis (ascending or descending).
multikey_sort    Sort the rows of a 2-D array by several columns.
topk             k largest / smallest values via ``np.argpartition``.
quickselect      k-th smallest element (educational implementation).
binary_search    Insertion index + existence flag (vectorised).
binary_search_py Pure-Python loop version of ``binary_search`` for one query.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stable_sort",
    "multikey_sort",
    "topk",
    "quickselect",
    "binary_search",
    "binary_search_py",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _to_array(values, name: str) -> np.ndarray:
    """Convert input to an ndarray and reject object arrays."""
    a = np.asarray(values)
    if a.dtype == object:
        raise TypeError(f"{name} must have a numeric or string dtype, got object")
    return a


def _check_int(k, name: str = "k") -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(k).__name__}")
    return int(k)


def _check_axis(a: np.ndarray, axis: int) -> int:
    if a.ndim == 0:
        raise ValueError("input must have at least 1 dimension, got a scalar")
    if not -a.ndim <= axis < a.ndim:
        raise np.exceptions.AxisError(axis, a.ndim)
    return axis % a.ndim


# --------------------------------------------------------------------------
# sorting
# --------------------------------------------------------------------------
def stable_sort(values, axis=-1, descending=False, return_indices=False):
    """Stable sort along an axis.

    Equal elements keep their original relative order in both ascending
    and descending mode.

    Parameters
    ----------
    values : array_like, shape (..., n, ...)
        Data to sort.
    axis : int or None, default -1
        Axis to sort along. ``None`` flattens the input first.
    descending : bool, default False
        Sort from largest to smallest.
    return_indices : bool, default False
        Also return the indices that sort ``values``.

    Returns
    -------
    sorted_values : ndarray, same shape as ``values`` (1-D if ``axis=None``)
    indices : ndarray of intp, same shape (only if ``return_indices``)

    Raises
    ------
    TypeError  If ``values`` has object dtype.
    AxisError  If ``axis`` is out of range.

    Notes
    -----
    NaNs are placed last in ascending order and first in descending order.
    Time O(n log n), extra space O(n).
    """
    a = _to_array(values, "values")
    if axis is None:
        a, axis = a.ravel(), 0
    axis = _check_axis(a, axis)
    if not descending:
        idx = np.argsort(a, axis=axis, kind="stable")
    else:
        # Stable descending: sort the reversed array, reverse the result and
        # map indices back. This keeps ties in their original order.
        n = a.shape[axis]
        rev_idx = np.argsort(np.flip(a, axis=axis), axis=axis, kind="stable")
        idx = (n - 1) - np.flip(rev_idx, axis=axis)
    out = np.take_along_axis(a, idx, axis=axis)
    return (out, idx) if return_indices else out


def multikey_sort(X, keys, descending=False, return_indices=False):
    """Sort the rows of a 2-D array by several columns.

    Parameters
    ----------
    X : array_like, shape (n_rows, n_cols)
    keys : sequence of int
        Column indices in priority order: ``keys[0]`` is the primary key,
        ``keys[1]`` breaks ties in ``keys[0]``, and so on.
    descending : bool or sequence of bool, default False
        Sort direction, either one value for all keys or one per key.
    return_indices : bool, default False
        Also return the row order.

    Returns
    -------
    X_sorted : ndarray, shape (n_rows, n_cols)
    order : ndarray of intp, shape (n_rows,) (only if ``return_indices``)

    Raises
    ------
    ValueError  If ``X`` is not 2-D, ``keys`` is empty, or the length of
                ``descending`` does not match ``keys``.
    IndexError  If a key is not a valid column index.

    Notes
    -----
    Each key column is first converted to integer codes with ``np.unique``
    (this works for numbers and strings, and lets us negate the codes for
    descending order without overflow). ``np.lexsort`` then sorts by all
    codes at once and is stable. The loop runs over the keys only.
    Time O(m * n log n) for m keys, space O(m * n).
    """
    A = _to_array(X, "X")
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D (n_rows, n_cols), got shape {A.shape}")
    keys = list(keys)
    if not keys:
        raise ValueError("keys must contain at least one column index")
    if isinstance(descending, (bool, np.bool_)):
        desc = [bool(descending)] * len(keys)
    else:
        desc = [bool(d) for d in descending]
        if len(desc) != len(keys):
            raise ValueError(
                f"descending has {len(desc)} entries but keys has {len(keys)}"
            )
    n_cols = A.shape[1]
    codes = []
    for col, d in zip(keys, desc):
        col = _check_int(col, "key")
        if not -n_cols <= col < n_cols:
            raise IndexError(f"key {col} out of range for {n_cols} columns")
        _, inv = np.unique(A[:, col], return_inverse=True)
        inv = inv.ravel().astype(np.int64)
        codes.append(-inv if d else inv)
    # np.lexsort treats the LAST key as primary, so reverse the list.
    order = np.lexsort(codes[::-1]) if A.shape[0] else np.empty(0, dtype=np.intp)
    out = A[order]
    return (out, order) if return_indices else out


# --------------------------------------------------------------------------
# top-k / selection
# --------------------------------------------------------------------------
def topk(values, k, largest=True, return_indices=True, axis=-1, sorted=True):
    """Return the k largest (or smallest) entries along an axis.

    Parameters
    ----------
    values : array_like, shape (..., n, ...)
    k : int, 0 <= k <= n
    largest : bool, default True
        Select the largest values; ``False`` selects the smallest.
    return_indices : bool, default True
        Also return the positions of the selected values.
    axis : int, default -1
    sorted : bool, default True
        Order the result (descending if ``largest``, else ascending).
        ``False`` skips this step and is faster.

    Returns
    -------
    top_values : ndarray, shape of ``values`` with ``n`` replaced by ``k``
    top_indices : ndarray of intp, same shape (only if ``return_indices``)

    Raises
    ------
    TypeError  If ``k`` is not an integer.
    ValueError If ``k`` is outside [0, n] or ``values`` is a scalar.
    AxisError  If ``axis`` is out of range.

    Notes
    -----
    NaNs are never preferred: they are treated as -inf when ``largest`` and
    +inf otherwise, so they are only returned if k exceeds the number of
    non-NaN values. Among tied values, which indices are chosen is not
    specified (same as ``np.argpartition``), but the returned values are
    always correct.
    Time O(n + k log k) per slice, space O(n).
    """
    a = _to_array(values, "values")
    axis = _check_axis(a, axis)
    k = _check_int(k)
    a_last = np.moveaxis(a, axis, -1)
    n = a_last.shape[-1]
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, {n}], got {k}")

    work = a_last
    if np.issubdtype(a.dtype, np.floating):
        nan_mask = np.isnan(a_last)
        if nan_mask.any():
            work = np.where(nan_mask, -np.inf if largest else np.inf, a_last)

    if k == 0:
        idx = np.empty(a_last.shape[:-1] + (0,), dtype=np.intp)
    else:
        if largest:
            idx = np.argpartition(work, n - k, axis=-1)[..., n - k:]
        else:
            idx = np.argpartition(work, k - 1, axis=-1)[..., :k]
        if sorted:
            sel = np.take_along_axis(work, idx, axis=-1)
            order = np.argsort(sel, axis=-1, kind="stable")
            if largest:
                order = order[..., ::-1]
            idx = np.take_along_axis(idx, order, axis=-1)

    vals = np.take_along_axis(a_last, idx, axis=-1)
    vals = np.moveaxis(vals, -1, axis)
    idx = np.moveaxis(idx, -1, axis)
    return (vals, idx) if return_indices else vals


def quickselect(values, k, seed=None):
    """Return the k-th smallest element (0-based) of a 1-D array.

    Educational implementation of Hoare's quickselect with a random pivot.
    Each round splits the data into ``< pivot``, ``== pivot`` and
    ``> pivot`` with vectorised boolean masks, then keeps only the part
    that contains the answer. The 3-way split makes duplicates and
    all-equal arrays cheap.

    Parameters
    ----------
    values : array_like, shape (n,)
    k : int, 0 <= k < n
        ``k=0`` gives the minimum, ``k=n-1`` the maximum.
    seed : int or None
        Seed for the pivot RNG (for reproducibility).

    Returns
    -------
    value : NumPy scalar with the dtype of ``values``
        Equal to ``np.sort(values)[k]``.

    Raises
    ------
    ValueError If ``values`` is not 1-D, is empty, contains NaN, or ``k``
               is out of range.
    TypeError  If ``k`` is not an integer.

    Notes
    -----
    Expected time O(n), worst case O(n^2). Extra space O(n).
    The input is never modified.
    """
    a = _to_array(values, "values")
    if a.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {a.shape}")
    n = a.size
    if n == 0:
        raise ValueError("quickselect on an empty array")
    k = _check_int(k)
    if not 0 <= k < n:
        raise ValueError(f"k must be in [0, {n - 1}], got {k}")
    if np.issubdtype(a.dtype, np.floating) and np.isnan(a).any():
        raise ValueError("values contains NaN; remove or impute NaNs first")

    rng = np.random.default_rng(seed)
    arr = a
    while True:
        pivot = arr[rng.integers(arr.size)]
        lows = arr[arr < pivot]
        if k < lows.size:
            arr = lows
            continue
        n_equal = np.count_nonzero(arr == pivot)
        if k < lows.size + n_equal:
            return pivot
        k -= lows.size + n_equal
        arr = arr[arr > pivot]


# --------------------------------------------------------------------------
# searching
# --------------------------------------------------------------------------
def binary_search(sorted_array, x, side="left", check_sorted=False):
    """Find where ``x`` would be inserted in a sorted array, and whether it exists.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Must be sorted ascending (NaNs, if any, at the end).
    x : scalar or array_like, any shape
        Value(s) to search for. Vectorised over ``x``.
    side : {'left', 'right'}, default 'left'
        'left' returns the first valid insertion index, 'right' the last.
    check_sorted : bool, default False
        Verify the array is sorted (costs O(n)).

    Returns
    -------
    index : int or ndarray of intp, same shape as ``x``
        Insertion index that keeps the array sorted.
    found : bool or ndarray of bool, same shape as ``x``
        Whether ``x`` is present in ``sorted_array``.

    Raises
    ------
    ValueError If ``sorted_array`` is not 1-D, ``side`` is invalid, or
               ``check_sorted`` is True and the array is unsorted.

    Notes
    -----
    Uses ``np.searchsorted`` (a C binary search).
    Time O(m log n) for m queries, space O(m).
    """
    a = _to_array(sorted_array, "sorted_array")
    if a.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {a.shape}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if check_sorted and a.size > 1 and np.any(a[1:] < a[:-1]):
        raise ValueError("sorted_array is not sorted in ascending order")

    xa = np.asarray(x)
    idx = np.searchsorted(a, xa, side=side)
    n = a.size
    if n == 0:
        found = np.zeros(xa.shape, dtype=bool)
    else:
        probe = idx if side == "left" else idx - 1
        cand = a[np.clip(probe, 0, n - 1)]
        found = cand == xa
        if np.issubdtype(a.dtype, np.floating) and np.issubdtype(xa.dtype, np.number):
            found = found | (np.isnan(cand) & np.isnan(xa))
        found = found & ((idx < n) if side == "left" else (idx > 0))
    if np.ndim(idx) == 0:
        return int(idx), bool(found)
    return idx, found


def binary_search_py(sorted_array, x):
    """Loop-based binary search for a single value (reference/benchmark version).

    Parameters
    ----------
    sorted_array : array_like, shape (n,), sorted ascending, no NaN
    x : scalar

    Returns
    -------
    index : int   Left insertion index (same as ``binary_search(..., side='left')``).
    found : bool  Whether ``x`` is present.

    Notes
    -----
    Classic half-open interval search on ``[lo, hi)``.
    Time O(log n), space O(1).
    """
    a = _to_array(sorted_array, "sorted_array")
    if a.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {a.shape}")
    lo, hi = 0, a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo, bool(lo < a.size and a[lo] == x)
