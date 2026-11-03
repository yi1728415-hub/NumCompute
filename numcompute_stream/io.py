"""CSV input for NumCompute-Stream.

Assignment 2.1 functions (unchanged): ``load_csv``, ``iter_csv``, ``CSVReader``.

Assignment 2.2 additions, built on the standard-library ``csv`` module and
NumPy only:

``load_dataset(path, target)``      whole file -> :class:`Dataset`
``stream_dataset(path, target, chunk_size)``  yields (X_num, X_cat, y) chunks
                                     straight from disk (bounded memory)
``iter_chunks(X, y, chunk_size)``    split arrays already in memory into chunks

Column types are inferred: a column whose non-empty cells all parse as numbers
is numeric (empty cells -> NaN); anything else is categorical text.
"""

import csv
from typing import NamedTuple

from io import StringIO
from pathlib import Path
from typing import Iterator, Union

import numpy as np


PathLike = Union[str, Path]


def _validate_path(filepath: PathLike) -> Path:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    return path


def _validate_options(delimiter: str, skip_header: int) -> None:
    if not isinstance(delimiter, str) or delimiter == "":
        raise ValueError("delimiter must be a non-empty string")
    if not isinstance(skip_header, int) or skip_header < 0:
        raise ValueError("skip_header must be a non-negative integer")


def load_csv(
    filepath: PathLike,
    delimiter: str = ",",
    dtype=float,
    missing_values=None,
    filling_values=np.nan,
    skip_header: int = 0,
) -> np.ndarray:

    """Load numeric or structured CSV data into a NumPy array.

    delimiter, dtype, missing_values, filling_values and skip_header are
    passed to numpy.genfromtxt. The returned shape follows that reader.
    Raises FileNotFoundError for a missing path and ValueError for invalid
    options or unparseable data. Time and storage are O(rows * columns)."""
    path = _validate_path(filepath)
    _validate_options(delimiter, skip_header)

    try:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            skip_header=skip_header,
        )

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse file '{path}': {exc}"
        ) from exc

    return np.asarray(data)


def iter_csv(
    filepath: PathLike,
    chunk_size: int,
    delimiter: str = ",",
    dtype=float,
    missing_values=None,
    filling_values=np.nan,
    skip_header: int = 0,
) -> Iterator[np.ndarray]:
    """Yield a CSV file in bounded-size NumPy chunks.

    Only ``chunk_size`` data rows are buffered at a time, so this is suitable
    for inputs that should not be loaded fully into memory.
    """

    path = _validate_path(filepath)
    _validate_options(delimiter, skip_header)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    def parse(lines: list[str]) -> np.ndarray:
        try:
            return np.asarray(
                np.genfromtxt(
                    StringIO("".join(lines)),
                    delimiter=delimiter,
                    dtype=dtype,
                    missing_values=missing_values,
                    filling_values=filling_values,
                    ndmin=2,
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not parse file '{path}': {exc}") from exc

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(skip_header):
            next(handle, None)
        rows: list[str] = []
        for line in handle:
            rows.append(line)
            if len(rows) == chunk_size:
                yield parse(rows)
                rows = []
        if rows:
            yield parse(rows)


class CSVReader:
    """Reusable CSV reader offering full and chunked reads."""

    def __init__(self, filepath: PathLike, **options):
        self.filepath = Path(filepath)
        self.options = dict(options)

    def read(self) -> np.ndarray:
        """Read the complete file using the options supplied at construction."""
        return load_csv(self.filepath, **self.options)

    def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield arrays containing at most chunk_size data rows at a time."""
        return iter_csv(self.filepath, chunk_size, **self.options)




# --------------------------------------------------------------------------
# Assignment 2.2: header-aware dataset loading and streaming
# --------------------------------------------------------------------------
class Dataset(NamedTuple):
    """A CSV split into numeric features, categorical features and the target.

    X_num : float ndarray (n_samples, n_numeric), NaN where a cell was empty
    X_cat : str ndarray (n_samples, n_categorical), "" where a cell was empty
    y     : ndarray (n_samples,), ints if the target column is integer-valued
    num_names, cat_names : list of str
    target : str
    """

    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray
    num_names: list
    cat_names: list
    target: str


_MISSING = {"", "na", "nan", "null", "none", "?"}


def _to_float(col):
    """Convert a column of strings to float; missing markers become NaN."""
    low = np.char.lower(np.char.strip(col.astype(str)))
    clean = np.where(np.isin(low, list(_MISSING)), "nan", low)
    return clean.astype(np.float64)


def _is_numeric(col):
    try:
        _to_float(col)
        return True
    except ValueError:
        return False


def _parse_target(col):
    try:
        vals = _to_float(col)
    except ValueError:
        return np.char.strip(col.astype(str))
    if np.isnan(vals).any():
        raise ValueError("the target column has missing values; drop those rows first")
    return vals.astype(np.int64) if np.all(vals == np.round(vals)) else vals


def _read_rows(path, delimiter):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"'{path}' is empty")
        yield [h.strip() for h in header]
        for line_no, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue                      # skip blank lines
            if len(row) != len(header):
                raise ValueError(f"'{path}' line {line_no}: expected {len(header)} "
                                 f"columns, got {len(row)}")
            yield row


def _split_columns(header, rows, target, numeric_cols):
    if target not in header:
        raise ValueError(f"target column {target!r} not found; columns are {header}")
    table = np.asarray(rows, dtype=str).reshape(len(rows), len(header))
    t = header.index(target)
    feats = [i for i in range(len(header)) if i != t]
    if numeric_cols is None:
        numeric_cols = [header[i] for i in feats if _is_numeric(table[:, i])]
    num_idx = [i for i in feats if header[i] in numeric_cols]
    cat_idx = [i for i in feats if header[i] not in numeric_cols]
    X_num = (np.column_stack([_to_float(table[:, i]) for i in num_idx])
             if num_idx else np.zeros((len(rows), 0)))
    X_cat = (np.char.strip(table[:, cat_idx]) if cat_idx
             else np.zeros((len(rows), 0), dtype=str))
    return (X_num, X_cat, _parse_target(table[:, t]),
            [header[i] for i in num_idx], [header[i] for i in cat_idx])


def load_dataset(filepath, target, delimiter=","):
    """Read a CSV with a header row into a :class:`Dataset`.

    Parameters
    ----------
    filepath : str or Path
    target : str
        Name of the label column.
    delimiter : str, default ","

    Raises
    ------
    FileNotFoundError, ValueError (missing target, ragged rows, empty file,
    missing labels).
    """
    path = _validate_path(filepath)
    _validate_options(delimiter, 0)
    rows = _read_rows(path, delimiter)
    header = next(rows)
    body = list(rows)
    if not body:
        raise ValueError(f"'{path}' has a header but no data rows")
    X_num, X_cat, y, num_names, cat_names = _split_columns(header, body, target, None)
    return Dataset(X_num, X_cat, y, num_names, cat_names, target)


def stream_dataset(filepath, target, chunk_size, delimiter=",", numeric_cols=None):
    """Yield ``(X_num, X_cat, y)`` chunks of at most ``chunk_size`` rows.

    Only one chunk is held in memory at a time. Column types are inferred from
    the first chunk unless ``numeric_cols`` (a list of column names) is given;
    the same split is then used for every later chunk, so all chunks have the
    same columns.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    path = _validate_path(filepath)
    _validate_options(delimiter, 0)
    rows = _read_rows(path, delimiter)
    header = next(rows)
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == chunk_size:
            X_num, X_cat, y, num_names, _ = _split_columns(header, batch, target, numeric_cols)
            numeric_cols = num_names if numeric_cols is None else numeric_cols
            yield X_num, X_cat, y
            batch = []
    if batch:
        X_num, X_cat, y, _, _ = _split_columns(header, batch, target, numeric_cols)
        yield X_num, X_cat, y


def iter_chunks(X, y, chunk_size, shuffle=False, random_state=None):
    """Yield ``(X_chunk, y_chunk)`` pairs of at most ``chunk_size`` rows.

    Order is preserved unless ``shuffle=True`` (one permutation up front).
    Slices are views, so no data is copied.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if shuffle:
        order = np.random.default_rng(random_state).permutation(X.shape[0])
        X, y = X[order], y[order]
    for start in range(0, X.shape[0], chunk_size):
        yield X[start:start + chunk_size], y[start:start + chunk_size]


__all__ = ["CSVReader", "Dataset", "iter_chunks", "iter_csv", "load_csv",
           "load_dataset", "stream_dataset"]
