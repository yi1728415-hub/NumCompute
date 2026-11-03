# NumCompute

A scientific computing toolkit built with Python and NumPy. It includes CSV
input, preprocessing, sorting and selection, ranking, descriptive statistics,
evaluation metrics, numerical derivatives and composable pipelines.

| Team member | Main responsibilities |
|---|---|
| Hongyu Liu | Data I/O, preprocessing, numerical utilities and video demonstration |
| Yi Su | Sorting, searching, ranking and statistics |
| Wenxuan Xue | Metrics, numerical differentiation, pipelines and integration |

## Installation and tests

Requires Python 3.10 or newer. The package's only runtime dependency is NumPy.
The optional development and demo dependencies provide pytest and Jupyter.
Create your own environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,demo]"
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` for the
remaining commands. Windows 11 has been tested; the other setup commands
have not been tested here.

## Quickstart

Run the complete example, including numerical assertions:

```powershell
.\.venv\Scripts\python.exe demo/integrated_quickstart.py
```

For the notebook, install the project kernel and open JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name python3 --display-name "Python (NumCompute)"
.\.venv\Scripts\python.exe -m jupyterlab demo/quickstart.ipynb
```

Select **Python (NumCompute)**, then restart the kernel and run all cells.
The notebook can be launched from the project root or `demo/`. It reads the
included six-row `demo/data/students.csv` and demonstrates missing values,
preprocessing, Pipeline fit/predict, tied ranks, streaming statistics,
classification metrics, gradients/Jacobian and benchmarks.

The classification metrics are illustrative in-sample results. The teaching
estimator thresholds the first feature at its fitted mean.

[Demo video](https://drive.google.com/file/d/10lPcQVu3qRCF2k28e1t6TSZvF5zuGlhQ/view?usp=sharing)

```python
import numpy as np
import numcompute as nc

X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 6.0]])
pipe = nc.make_pipeline(nc.Imputer(fill_value=0.0), nc.StandardScaler())
prepared = pipe.fit_transform(X)
new_prepared = pipe.transform([[3.0, 4.0]])
largest = nc.topk(prepared[:, 0], 2, return_indices=False)
ranks = nc.rank_values([60, 78, 85, 92, 92, 95], method="average")
gradient = nc.gradient(lambda z: np.sum(z**2), [1.0, 2.0])
```

## API and conventions

| Module / owner | Main API | Input/output conventions |
|---|---|---|
| io / Hongyu Liu | load_csv, CSVReader, iter_csv | NumPy arrays; delimiter/dtype/missing-value options; chunk_size bounds buffered CSV rows |
| preprocessing / Hongyu Liu | Imputer (SimpleImputer), StandardScaler, MinMaxScaler, OneHotEncoder | 2-D `(samples, features)`; fit returns self; transform preserves rows; one-hot may expand columns |
| utils / Hongyu Liu | softmax, logsumexp, sigmoid, relu, euclidean_distance, batch_iterator, topk_indices | Activations/distances are vectorised; softmax defaults to last axis; batch_iterator yields row batches |
| sort_search / Yi Su | stable_sort, multikey_sort, topk, quickselect, binary_search | topk returns `(values, indices)` unless return_indices=False; quickselect k is zero-based; binary_search returns insertion index and existence |
| rank / Yi Su | rank_values, percentile | Ascending ranks start at 1; average/dense/ordinal tie methods; percentile q in `[0,100]` |
| stats / Yi Su | mean, median, var, std, describe, histogram, quantile, RunningStats | axis=0 reduces rows; nan_policy controls NaNs where supported; quantile q in `[0,1]`; var/std default ddof=0, describe defaults ddof=1 |
| metrics / Wenxuan Xue | accuracy, precision, recall, f1, mse, confusion_matrix, roc_curve, roc_auc_score | 1-D matching label arrays; confusion rows=true, columns=predicted; regression arrays must match shape |
| optim / Wenxuan Xue | gradient, jacobian | Scalar function gradient has x.shape; Jacobian shape is F(x).shape + x.shape; central differences by default |
| pipeline / Wenxuan Xue | Pipeline, make_pipeline, FeatureUnion, make_union | Intermediate steps implement fit/transform; final estimator may implement predict; FeatureUnion concatenates feature columns |
| benchmarking / Wenxuan Xue | benchmark, benchmark_many, format_results | Per-call timings in seconds; configurable repeat, number and warmup |

`nc.rank` is the module; call `nc.rank_values(...)` or `nc.rank.rank(...)`.
Statistics reduce over rows with `axis=0`; `axis=None` reduces all values.
Quantiles use fractions in [0,1], while percentiles use [0,100]. The default
variance is the population variance (`ddof=0`); `describe` defaults to `ddof=1`.
Impute missing features before scaling, and reuse fitted preprocessors for
new data rather than fitting them again.

## Benchmarks

```powershell
.\.venv\Scripts\python.exe benchmark/bench_integrated.py
.\.venv\Scripts\python.exe benchmark/bench_member_b.py
.\.venv\Scripts\python.exe benchmark/benchmarking.py
```

The integrated comparison below was measured on 17 October 2026 with
Windows 11, CPython 3.14.6, NumPy 2.5.4 and an Intel CPU with 20 logical cores.
It checks matching results before timing, uses seed 42, and reports median
time per call over five repeats with three calls per repeat and one warmup.
Input/list preparation, scaler fitting and CSV I/O are outside the timing.

| Task | Data size | Python baseline (ms) | NumCompute (ms) | Speed-up | Same result |
|---|---|---:|---:|---:|:---:|
| StandardScaler.transform | 20,000 x 4 | 7.4222 | 0.5024 | 14.77x | yes |
| stable softmax | 20,000 x 4 | 12.4510 | 1.5772 | 7.89x | yes |
| top-k | n=20,000, k=10 | 1.9376 | 0.0462 | 41.94x | yes |
| accuracy | n=20,000 | 0.5824 | 0.0204 | 28.50x | yes |

The top-k baseline performs a full Python sort; NumCompute uses partial
selection. That comparison includes an algorithm difference as well as
vectorisation. Both softmax implementations subtract the row maximum.

The saved measurements and environment are in
[results_integrated.md](benchmark/results_integrated.md).
Yi Su's larger seven-task comparison is in
[results_yi_su.md](benchmark/results_yi_su.md), produced by `bench_member_b.py`.
That top-k comparison uses `heapq.nlargest` on one million values, so it has
a different baseline and input size from the integrated table and video.
The video and the saved tables were measured in separate runs. Times and
speed ratios vary with the environment; the numerical results are checked.

## Project layout

```text
numcompute/                 # ten modules and the public API
tests/                      # 122 unit and integration cases
benchmark/                  # comparison scripts and saved measurements
demo/
  data/students.csv         # synthetic CSV input
  quickstart.ipynb          # notebook with saved outputs
  integrated_quickstart.py  # executable example
  build_notebook.py         # notebook generation from the example sections
  verify_demo.py            # fresh-kernel checks from root and demo/
  demo_yi_su.ipynb          # algorithms/statistics examples
  demo_wenxuan_xue.ipynb     # metrics/optimisation/pipeline examples
pyproject.toml
```

To check the notebook in fresh kernels:

```powershell
.\.venv\Scripts\python.exe demo/verify_demo.py
```

When changing the demo source, regenerate the notebook with
`python demo/build_notebook.py`, then run `python demo/verify_demo.py --write`
to verify it and save the outputs.
