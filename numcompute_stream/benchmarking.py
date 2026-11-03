"""Reusable micro-benchmark harness based on :func:`time.perf_counter`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary for one callable.

    ``durations`` contains seconds per function call, one value for each
    repeat. ``value`` is the return value from the final measured invocation.
    """

    name: str
    repeat: int
    number: int
    durations: tuple[float, ...]
    value: Any = None

    @property
    def calls(self) -> int:
        """Return the total number of timed calls, excluding warmup."""
        return self.repeat * self.number

    @property
    def minimum(self) -> float:
        """Return the smallest measured duration per call in seconds."""
        return min(self.durations)

    @property
    def maximum(self) -> float:
        """Return the largest measured duration per call in seconds."""
        return max(self.durations)

    @property
    def mean(self) -> float:
        """Return the mean measured duration per call in seconds."""
        return mean(self.durations)

    @property
    def median(self) -> float:
        """Return the median measured duration per call in seconds."""
        return median(self.durations)

    @property
    def std(self) -> float:
        """Return the population standard deviation of per-call durations."""
        return pstdev(self.durations)

    def as_dict(self) -> dict[str, Any]:
        """Return the timing settings and summary statistics as a dictionary."""
        return {
            "name": self.name,
            "calls": self.calls,
            "repeat": self.repeat,
            "number": self.number,
            "min_seconds": self.minimum,
            "mean_seconds": self.mean,
            "median_seconds": self.median,
            "max_seconds": self.maximum,
            "std_seconds": self.std,
        }


def benchmark(
    function: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    repeat: int = 5,
    number: int = 1,
    warmup: int = 1,
    kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Benchmark a callable and return per-call timing statistics."""

    if not callable(function):
        raise TypeError("function must be callable")
    for option_name, option_value, minimum in (
        ("repeat", repeat, 1),
        ("number", number, 1),
        ("warmup", warmup, 0),
    ):
        if not isinstance(option_value, int) or option_value < minimum:
            raise ValueError(f"{option_name} must be an integer >= {minimum}")
    call_kwargs = dict(kwargs or {})

    for _ in range(warmup):
        function(*args, **call_kwargs)

    samples: list[float] = []
    last_value: Any = None
    for _ in range(repeat):
        started = perf_counter()
        for _ in range(number):
            last_value = function(*args, **call_kwargs)
        samples.append((perf_counter() - started) / number)

    result_name = name or getattr(function, "__name__", function.__class__.__name__)
    return BenchmarkResult(result_name, repeat, number, tuple(samples), last_value)


def benchmark_many(
    cases: Mapping[str, Callable[[], Any]],
    *,
    repeat: int = 5,
    number: int = 1,
    warmup: int = 1,
) -> list[BenchmarkResult]:
    """Benchmark named zero-argument callables in mapping order."""

    if not isinstance(cases, Mapping) or not cases:
        raise ValueError("cases must be a non-empty mapping")
    return [
        benchmark(
            function,
            name=name,
            repeat=repeat,
            number=number,
            warmup=warmup,
        )
        for name, function in cases.items()
    ]


def format_results(results: Sequence[BenchmarkResult], *, unit: str = "ms") -> str:
    """Format results as a compact, dependency-free text table."""

    factors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if unit not in factors:
        raise ValueError("unit must be one of 's', 'ms', 'us', or 'ns'")
    if not results:
        return ""
    scale = factors[unit]
    header = f"{'name':<24} {'min':>12} {'mean':>12} {'median':>12} {'max':>12} {'std':>12}"
    rows = [header, "-" * len(header)]
    for result in results:
        rows.append(
            f"{result.name:<24} "
            f"{result.minimum * scale:>10.4f} {unit:<2} "
            f"{result.mean * scale:>10.4f} {unit:<2} "
            f"{result.median * scale:>10.4f} {unit:<2} "
            f"{result.maximum * scale:>10.4f} {unit:<2} "
            f"{result.std * scale:>10.4f} {unit:<2}"
        )
    return "\n".join(rows)


__all__ = ["BenchmarkResult", "benchmark", "benchmark_many", "format_results"]
