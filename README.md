# NumCompute

> **A modular scientific computing & streaming decision tree toolkit in Plain Python and NumPy only.**  
> Built for *Programming for AI (Assignments 2.1 & 2.2)* at the University of Adelaide.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-pure%20numpy-orange.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-400%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-pep8-blue.svg)]()

---

## 1. Project Overview & Philosophy

**NumCompute** implements core scientific computing and streaming machine learning primitives from first principles using **only plain Python and NumPy** (strictly no pandas, scikit-learn, PyTorch, or SciPy).

The repository contains two cohesive packages:
1. **`numcompute` (Assignment 2.1)**: Core array algorithms, descriptive statistics (Welford), sorting/ranking, numerical optimization (line search, finite-difference Jacobians), and scikit-learn compatible batch preprocessing/pipelines.
2. **`numcompute_stream` (Assignment 2.2)**: A high-throughput streaming decision tree framework featuring vectorised split evaluations, transactional buffered retraining, the Transformer Snapshot Pattern, strict prequential evaluation (test-then-train), online metrics, and chunked CSV streaming.

### Core Technical Pillars

1. **Vectorisation First**: Core computations avoid explicit Python loops, utilising NumPy broadcasting, universal functions (`ufuncs`), contiguous slicing, and vectorized indexing for speed-ups ranging from **6x to over 900x**.
2. **Transformer Snapshot Pattern**: Solves feature coordinate drift in streaming pipelines by freezing intermediate transformer projections at model retraining time, ensuring 100% consistent feature coordinates during inference.
3. **Prequential Integrity**: Strict test-then-train protocol via `StreamTrainer` that evaluates every incoming chunk against the existing model before updating weights, preventing data leakage.
4. **Numerical Stability**:
   - Max-shifted $\log\sum\exp$ and Softmax formulations preventing overflow and underflow.
   - Welford's single-pass algorithm and Chan's parallel merge for online mean/variance updates in strict $O(d)$ space.
   - Overflow-safe midpoint calculations: $a + 0.5 \cdot (b - a)$.
5. **Transactional Failure Atomicity**: Retraining operations use candidate swapping; any runtime error during candidate training leaves prior models, classes, and pending buffers completely intact.
6. **Rigorous Quality Assurance**: **400 automated tests** passing under `pytest -W error` with zero warnings, covering all edge cases, numerical boundaries, and 6 end-to-end integration scenarios.

---

## 2. Directory Structure

```text
NumCompute/
├── numcompute/                     # Core scientific computing library (Assignment 2.1)
│   ├── __init__.py                 # Top-level exports and versioning
│   ├── io.py                       # CSV/TSV reader, chunked streaming, missing value handling
│   ├── preprocessing.py            # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── sort_search.py              # Stable sorts, top-k (argpartition), quickselect, binary search
│   ├── rank.py                     # Ranking with ties (average, dense, ordinal), percentiles
│   ├── stats.py                    # Descriptive stats, quantiles, histograms, streaming Welford
│   ├── metrics.py                  # Accuracy, precision, recall, f1, confusion matrix, MSE, ROC/AUC
│   ├── optim.py                    # Finite differences (grad, jacobian), gradient checking, line search
│   ├── pipeline.py                 # Transformer/Estimator API, Pipeline, Compose, FeatureUnion
│   ├── utils.py                    # Numerical activations, stable softmax/logsumexp, pairwise distances
│   └── benchmarking.py             # Micro-benchmarking harness and environment inspection
├── numcompute_stream/              # Streaming Decision Tree Framework (Assignment 2.2)
│   ├── __init__.py                 # Public streaming API exports
│   ├── tree.py                     # DecisionTreeClassifier with vectorised split search & buffered retraining
│   ├── preprocessing.py            # Online StandardScaler (Chan), Imputer, OneHotEncoder
│   ├── stats.py                    # StreamingStats (online moments, histograms, quantiles)
│   ├── metrics.py                  # StreamingMetrics (confusion matrix, macro/binary F1, AUC) & RollingMetrics
│   ├── pipeline.py                 # Streaming Pipeline with Transformer Snapshot Pattern & raw buffer
│   ├── stream.py                   # StreamTrainer (prequential test-then-train coordinator, memory proxy)
│   ├── io.py                       # Pure-Python CSVReader (RFC-4180 parsing, chunk generator)
│   └── visualise.py                # Reusable Matplotlib visualisations (metric trajectory, model comparison)
├── tests/                          # 400 automated pytest tests (281 baseline + 119 streaming)
│   ├── test_tree.py                # Decision tree split evaluation & buffered retraining tests
│   ├── test_preprocessing_streaming.py # Online transformer robustness & edge-case tests
│   ├── test_stats_streaming.py     # StreamingStats moments & histogram tests
│   ├── test_metrics_streaming.py   # Atomic StreamingMetrics & RollingMetrics tests
│   ├── test_pipeline_streaming.py  # Pipeline snapshot pattern & coordinate consistency tests
│   ├── test_stream.py              # StreamTrainer prequential protocol tests
│   ├── test_io_streaming.py        # CSVReader RFC-4180 parsing & chunk iteration tests
│   ├── test_visualise.py           # Matplotlib visualisation validation tests
│   ├── test_integration_streaming.py # 6 mandatory end-to-end integration scenarios
│   └── test_*.py                   # 281 original baseline tests
├── demo/                           # Interactive demonstration notebooks and datasets
│   ├── stream_demo.ipynb           # Fully executed streaming quickstart notebook (Assignment 2.2)
│   ├── stream_demo_dataset.csv     # 1,200-sample streaming CSV fixture with missing values
│   ├── quickstart.ipynb            # Batch core demonstration notebook (Assignment 2.1)
│   ├── sample_data.csv             # 200-row batch CSV dataset
│   └── sample_data.tsv             # Tab-separated dataset
├── benchmark/                      # Performance benchmarking suite
│   ├── benchmark_streaming.py      # Split search (loop vs vec) & buffer-size tradeoff benchmarks
│   ├── results/                    # Generated benchmark outputs
│   │   ├── results_streaming.md    # Benchmark markdown tables and performance analysis
│   │   ├── streaming_benchmark_results.json # Raw benchmark metric JSON
│   │   ├── loop_vs_vectorized.png  # Latency and speedup plot
│   │   └── buffer_size_tradeoff.png# Throughput, latency, and accuracy tradeoff plot
│   ├── run_benchmarks.py           # Assignment 2.1 core benchmarking script
│   └── results.md                  # Assignment 2.1 benchmark summary
├── report/                         # Technical report
│   └── report.md                   # Formal technical report structured for <= 4-page PDF export
├── pyproject.toml                  # Packaging configuration (supports editable install)
├── BUGLOG.md                       # Structured defect tracking & regression tests
├── CONVENTIONS.md                  # Architectural conventions and design principles
└── README.md                       # Project documentation & usage guides
```

---

## 3. Quickstart: Streaming Machine Learning

```python
from numcompute_stream.io import CSVReader
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer

# 1. Initialize lazy CSV stream (RFC-4180 compliant)
reader = CSVReader("demo/stream_demo_dataset.csv", target_column="target")

# 2. Build pipeline with Transformer Snapshot Pattern (avoids coordinate drift)
pipe = Pipeline([
    ("imputer", Imputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(buffer_size=150, max_depth=5, random_state=42)),
], buffer_size=150)

# 3. Stream with strict prequential test-then-train evaluation
trainer = StreamTrainer(pipe)

for X_chunk, y_chunk in reader.iter_chunks(chunk_size=75):
    log = trainer.fit_chunk(X_chunk, y_chunk)
    print(f"Chunk {log['chunk_index']}: Acc={log['chunk_accuracy']}, "
          f"CumulAcc={log['cumulative_accuracy']:.3f}, Retrained={log['retrained']}")
```

---

## 4. Installation & Setup

NumCompute requires **Python 3.9+** and **NumPy 1.24+** (and `matplotlib >= 3.5` for visualisations).

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/DivyViradiya/NumCompute.git
cd NumCompute

python3 -m venv ~/numcompute-venv
source ~/numcompute-venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Step 2: Run the Unit & Integration Tests

Execute the complete 400-test suite with zero warnings allowed:

```bash
python -m pytest -W error tests -q
```
Expected output:
```text
400 passed in ~0.97s
```

Run doctest verification across all modules:
```bash
python -m pytest --doctest-modules numcompute numcompute_stream
```

### Step 3: Run Streaming Benchmarks

Execute the automated benchmark harness:
```bash
python benchmark/benchmark_streaming.py
```
This generates:
- `benchmark/results/results_streaming.md`
- `benchmark/results/streaming_benchmark_results.json`
- `benchmark/results/loop_vs_vectorized.png`
- `benchmark/results/buffer_size_tradeoff.png`
```bash
python -m pytest tests -W error -q
python -m doctest numcompute/*.py
```

---

## 4. Quickstart Examples

### 4.1 Ingestion & Missing Data (`numcompute.io`)

```python
import numcompute.io as nio

# Load numeric columns, skipping headers and filling NaNs with -1.0
X_filled = nio.load_csv("demo/sample_data.csv", skip_header=1, usecols=[0, 1], missing="fill", fill_value=-1.0)

# Load mixed numeric and categorical data in one call
X_num, X_cat, num_cols, cat_cols = nio.load_mixed_csv(
    "demo/sample_data.csv", categorical_cols=["city", "plan"]
)

# Stream massive CSV files in chunks without loading entirely into memory
for chunk in nio.iter_csv_chunks("demo/sample_data.csv", chunksize=50, skip_header=1, usecols=[0, 1]):
    print(f"Processed batch chunk with shape {chunk.shape}")
```

### 4.2 Data Preprocessing & Pipelines (`numcompute.preprocessing`, `numcompute.pipeline`)

```python
import numpy as np
from numcompute.preprocessing import Imputer, StandardScaler, OneHotEncoder
from numcompute.pipeline import Pipeline, ColumnSelector, FeatureUnion

# Sample raw feature matrix: [Age, Income]
X_raw = np.array([
    [25.0, 50000.0],
    [np.nan, 65000.0],
    [38.0, np.nan],
    [45.0, 80000.0]
])

# Build an end-to-end preprocessing pipeline
preprocessor = Pipeline([
    ("impute", Imputer(strategy="median")),
    ("scale", StandardScaler())
])

X_scaled = preprocessor.fit_transform(X_raw)
print("Standardised features:\n", X_scaled)
```

### 4.3 High-Performance Sorting, Searching & Ranking (`numcompute.sort_search`, `numcompute.rank`)

```python
from numcompute import sort_search, rank

data = np.array([12.0, 45.0, 3.0, 45.0, 99.0, 21.0])

# Top-k largest elements in O(n) using partial partition
top_vals, top_idx = sort_search.topk(data, k=3, largest=True)
print("Top 3 values:", top_vals, "Indices:", top_idx)

# Quickselect for k-th smallest element without full sort
median_val = sort_search.quickselect(data, k=len(data) // 2)

# Binary search returning (insertion_index, exists_flag)
idx, found = sort_search.binary_search(np.sort(data), 45.0)

# Ranking with tie handling ('average', 'dense', 'ordinal', 'min', 'max')
avg_ranks = rank.rank(data, method="average")
print("Average ranks with ties handled:", avg_ranks)
```

### 4.4 Descriptive & Streaming Statistics (`numcompute.stats`)

```python
from numcompute import stats

# NaN-safe statistics
X = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
print("Mean (omitting NaN):", stats.mean(X, nan_policy="omit"))
print("Histogram bins:", stats.histogram(X, bins=4))

# Numerically stable streaming Welford statistics
running = stats.RunningStats()
for batch in [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]:
    running.update(batch)

print("Streaming Mean:", running.mean, "| Variance:", running.variance)
```

### 4.5 Numerical Optimization (`numcompute.optim`)

```python
from numcompute import optim

# Define scalar objective function: f(x) = x0^2 + 3*x1 + sin(x0 * x1)
f = lambda x: x[0]**2 + 3*x[1] + np.sin(x[0] * x[1])
x0 = np.array([1.0, 2.0])

# Finite-difference gradient (central differences, O(h^2) truncation error)
gradient = optim.grad(f, x0, method="central", h=1e-5)
print("Estimated gradient:", gradient)

# Vector Jacobian F: R^2 -> R^3
F = lambda x: np.array([x[0] * x[1], x[0] + x[1]**2, np.sin(x[0])])
J = optim.jacobian(F, x0)
print("Jacobian matrix shape:", J.shape)
```

### 4.6 Evaluation Metrics (`numcompute.metrics`)

```python
from numcompute import metrics

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3])

print("Accuracy :", metrics.accuracy(y_true, y_pred))
print("F1-Score :", metrics.f1(y_true, y_pred, average="binary"))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_true, y_pred))

# Bonus ROC-AUC
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC Score:", roc_auc)
```

---

## 5. Complete API Overview

| Module | Primary Classes & Functions | Purpose & Algorithmic Notes |
| :--- | :--- | :--- |
| **`numcompute.io`** | `load_csv`, `load_mixed_csv`, `iter_csv_chunks`, `read_header` | High-throughput data ingestion via `np.genfromtxt`, delimiter detection, chunked streaming generators, and missing value strategies (`'nan'`, `'fill'`, `'skip'`). |
| **`numcompute.preprocessing`**| `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder` | Scikit-learn protocol transformers. Z-score normalization, min-max feature range scaling, mean/median/constant imputation, and categorical one-hot encoding with unknown value handling. |
| **`numcompute.sort_search`** | `stable_sort`, `stable_argsort`, `multi_key_sort`, `topk`, `quickselect`, `binary_search` | Sorting and searching algorithms. Fast $O(n)$ top-$k$ selection using `np.argpartition`, Lomuto/Hoare partition quickselect, and binary search with boundary insertion semantics. |
| **`numcompute.rank`** | `rank`, `percentile` | Fractional and discrete ranking supporting `'average'`, `'min'`, `'max'`, `'dense'`, and `'ordinal'` tie breaking; multi-quantile percentiles (`linear`, `lower`, `higher`, `midpoint`, `nearest`). |
| **`numcompute.stats`** | `mean`, `median`, `var`, `std`, `minimum`, `maximum`, `histogram`, `quantile`, `RunningStats` | Comprehensive summary statistics with axis management and `nan_policy` control. Online single-pass Welford algorithm and Chan's parallel batch merge for variance. |
| **`numcompute.metrics`** | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `rmse`, `mae`, `r2_score`, `roc_curve`, `auc`, `roc_auc_score` | Binary and multi-class classification metrics (macro, micro, weighted), regression loss metrics, and trapezoidal numerical ROC-AUC. |
| **`numcompute.optim`** | `grad`, `jacobian`, `gradient_check`, `backtracking_line_search` | First-order numerical calculus. Forward, backward, and central finite differences ($O(h^2)$ precision), relative-step scaling, and Armijo backtracking line search. |
| **`numcompute.pipeline`** | `Pipeline`, `Compose`, `FeatureUnion`, `ColumnSelector`, `FunctionTransformer`, `NearestCentroidClassifier` | Estimator/Transformer chaining protocol with `fit`, `transform`, `fit_transform`, and `predict`. Parameter inspection via `get_params` and `set_params`. |
| **`numcompute.utils`** | `softmax`, `log_softmax`, `logsumexp`, `sigmoid`, `relu`, `leaky_relu`, `tanh`, `softplus`, `pairwise_distances`, `iter_batches` | Numerically stable neural primitives with max-shift correction, vectorised distance metrics (Euclidean, Manhattan, Chebyshev, Cosine), and array validation helpers. |
| **`numcompute.benchmarking`**| `BenchmarkResult`, `time_function`, `compare`, `format_table`, `environment_info` | Micro-benchmarking test harness measuring wall-clock latency, throughput, memory strides, and Markdown comparison tables. |

---

## 6. Performance Benchmarks

NumCompute was micro-benchmarked by comparing vectorised NumPy implementations against standard Python loops performing identical numerical operations across multiple problem dimensions.

### Benchmark Environment
- **Platform**: Windows-11-AMD64 (20 logical cores, Intel64 Family 6 Model 186) & macOS-ARM64 (Apple Silicon)
- **Python**: 3.14 / 3.15
- **NumPy**: 2.4.0 / 2.5.4
- **Timing Protocol**: Best of 3 runs, reproducible random seed (`seed=0`).

### Performance Summary Table (Extracted from `benchmark/results.md`)

| Operation | Benchmark Size | Python Loop | Vectorised NumCompute | Measured Speed-Up | Outputs Match |
| :--- | ---:| ---:| ---:| ---:| :---: |
| **Standardise Features (10 cols)** | 20,000 rows | 92.30 ms | 2.80 ms | **33.0x** | Yes |
| **Row-wise Softmax (20 classes)** | 20,000 rows | 74.41 ms | 5.24 ms | **14.2x** | Yes |
| **Pairwise Euclidean Distances (8 feats)** | 400 points | 209.58 ms | 1.77 ms | **118.2x** | Yes |
| **Average Ranks with Ties** | 2,000 values | 164.44 ms | 176.1 µs | **933.8x** | Yes |
| **Top-10 Selection** | 300,000 values | 194.32 ms | 3.46 ms | **56.2x** | Yes |
| **Histogram Construction (20 bins)** | 500,000 values | 63.22 ms | 10.33 ms | **6.1x** | Yes |
| **Streaming Welford Mean/Variance** | 500,000 values | 37.98 ms | 4.08 ms | **9.3x** | Yes |
| **Finite-Difference Gradient (Batched)** | 100 points | 550.2 µs | 32.3 µs | **17.0x** | Yes |

*Note: In `pairwise_distances` and `rank`, vectorised broadcasting and sorting algorithms bypass Python pointer chasing, resulting in speed-ups between **118x and 933x**. In batched finite differences at very large scale ($n \ge 1,500$), function evaluation dominates memory traffic, converging to ~1.1x–1.3x.*

---

## 7. Testing & Quality Assurance

NumCompute enforces rigorous automated testing. The test suite comprises **281 unit tests** with complete pass rates:

```bash
python -m pytest tests -q
# ........................................................................ [ 25%]
# ........................................................................ [ 51%]
# ........................................................................ [ 76%]
# .................................................................        [100%]
# 281 passed in 0.35s
```

### Edge Cases Covered
- **Empty & Degenerate Arrays**: `(0,)`, `(0, 0)`, `(0, 5)` inputs tested across `topk`, `one_hot`, `pairwise_distances`, `histogram`, and `StandardScaler`.
- **Constant & All-Equal Arrays**: Zero-variance columns in `StandardScaler` (scales to zero without `NaN`), identical inputs in `MinMaxScaler` mapping to lower bounds.
- **Tied Values**: Extensive tie checking in `topk`, `rank` (`average`, `dense`, `ordinal`), and `binary_search` (`bisect_left` vs `bisect_right`).
- **Extreme $k$**: $k = 0$, $k = 1$, $k = n$, and $k > n$ clamped safely without buffer overflows.
- **Non-Contiguous Strides**: Slices with step sizes (e.g., `X[::-1]`, `X[::2, ::-1]`) tested across all sorting, searching, and preprocessing functions.
- **NaN Handling**: Comprehensive verification of `nan_policy="propagate"`, `"omit"`, and `"raise"`.
- **Mutation Checks**: Deliberate single-line mutations (e.g., modifying Welford merge constants, altering average rank formulas, tampering with F1 numerators) confirmed to trigger test suite failures.

---

## 8. Interactive Demo Notebook

An end-to-end demonstration is provided in [demo/quickstart.ipynb](demo/quickstart.ipynb), executable via Jupyter:

```bash
jupyter notebook demo/quickstart.ipynb
```

The notebook illustrates:
1. Loading raw CSV and TSV files with missing tokens using `io.py`.
2. Constructing an end-to-end classification pipeline with `Imputer`, `StandardScaler`, and `NearestCentroidClassifier`.
3. Evaluating predictions with `accuracy`, `f1`, `confusion_matrix`, and `roc_curve`.
4. High-performance `topk`, `quickselect`, `binary_search`, and rank tie handling.
5. Computing summary statistics, text histograms, and chunked streaming Welford stats.
6. Gradient estimation, step-size truncation analysis, and Jacobian calculations.
7. Micro-benchmarking table comparing Python loops against vectorised NumCompute operations.

---

## 9. Team Contributions & Division of Work

In accordance with course academic integrity guidelines, all member contributions across **code authoring**, **written report**, and **video presentation** deliverables are transparently recorded below.

### 9.1 Deliverables & Responsibility Matrix

| Project Deliverable | Primary Contributor(s) | Key Deliverables Produced |
| :--- | :--- | :--- |
| **Codebase & Architecture** | **Divy Viradiya** (Lead) | Complete modular architecture, 10 core modules, 281 unit tests, `CONVENTIONS.md`, `demo/quickstart.ipynb`. |
| **Batch Integration & QA** | **Mihir Patel**, **Hetsi Patel**, **Tanmay Patel** | Branch review, cross-platform validation, bug reporting (`BUGLOG.md`), benchmark verification. |
| **Written Technical Report** | **Tanmay Patel** & **Hetsi Patel** | Comprehensive assignment report, methodology write-up, experimental analysis, and formatting. |
| **Video Production & Script**| **Mihir Patel** | Video production planning, script drafting (`VIDEO_SCRIPT.pdf`), slide/screen assets, and final video editing. |
| **Video Screen Recording** | **Divy Viradiya** | Solo screen capture walkthrough, code & notebook demonstration, and spoken voiceover recording. |

---

### 9.2 Detailed Member Breakdown

| Member Name | GitHub / Git Handle | Role & Deliverable Contributions | Merged Branches / Artifacts |
| :--- | :--- | :--- | :--- |
| **Divy Viradiya** | [`@DivyViradiya`](https://github.com/DivyViradiya) | **Lead Architect & Video Presenter**: Designed architecture and authored all core modules (`utils`, `io`, `preprocessing`, `sort_search`, `rank`, `stats`, `metrics`, `optim`, `pipeline`, `benchmarking`), test suites (281 tests), `demo/quickstart.ipynb`, `README.md`, resolved core defects (BUG-01, BUG-03). Conducted the **live video screen recording and full spoken voiceover**. | `main`, PR #2 (`feat/core-modules`), commit `ebf5a33`, `128ef59`, `VIDEO_SCRIPT.md` |
| **Mihir Patel** | [`@mihirpatel2409`](https://github.com/mihirpatel2409) | **Batch 3 Integration & Video Producer**: Integrated Batch 3 (`sort_search.py`, `rank.py`, and test suites). Verified edge cases for top-$k$, quickselect, and ranking ties. Assisted with `BUGLOG.md`. Co-authored the **video script structure and completed final video editing / audio mastering**. | PR #1 (`feat/sort-search-rank`), commit `11da362`, merge `b31cc16`, `VIDEO_SCRIPT.pdf` |
| **Hetsi Patel** | [`@Hetsi1910`](https://github.com/Hetsi1910) | **Batch 4 Integration & Report Co-Author**: Integrated and verified Batch 4 (`stats.py`, `metrics.py`, and test suites). Diagnosed and documented import dependency ordering issue on `main` (PROC-01 in `BUGLOG.md`). Validated ROC/AUC calculations. Co-authored and formatted the **final written technical assignment report**. | PR #3 (`feat/stats-metrics`), commit `fcbe3ab`, merge `b67f454`, Final Assignment Report |
| **Tanmay Patel** | [`@Legacy164`](https://github.com/Legacy164) | **Batch 5 Integration & Report Co-Author**: Integrated Batch 5 (`optim.py`, `benchmarking.py`, final package init, and test suites). Executed complete benchmark suite on Windows hardware (`benchmark/results.md`). Validated finite-difference Jacobians. Co-authored the **final written technical assignment report** and performance discussions. | PR #4 (`feat/optim-benchmark-final`), commit `728a478`, merge `60ae707`, Final Assignment Report |

---

## 10. License

Developed for educational purposes as part of the ADELAIDE UNIVERSITY *Programming for AI* curriculum. Released under the MIT License.
