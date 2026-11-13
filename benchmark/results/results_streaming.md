# NumCompute Streaming Performance & Empirical Evaluation

**Environment:** Python 3.15.0, NumPy 2.5.4 (darwin)
**Date:** 2026-11-13 17:37:45

## 1. Vectorised vs. Naive Python Loop Split Evaluation

Evaluated on synthetic multi-class datasets ($C=3$) with 2 warm-up iterations and 10 measured repetitions.
Timings report both median and mean execution times.

| Sample Size (N) | Naive Loop Median (ms) | Vectorised Median (ms) | Speedup (Median) | Gain Δ | Threshold Δ | Partition Match |
|---|---|---|---|---|---|---|
| 100 | 0.418 | 0.059 | **7.0x** | 0.00e+00 | 0.00e+00 | Yes (Exact) |
| 500 | 2.116 | 0.213 | **10.0x** | 5.55e-17 | 0.00e+00 | Yes (Exact) |
| 1000 | 4.299 | 0.419 | **10.3x** | 2.78e-17 | 0.00e+00 | Yes (Exact) |
| 2000 | 8.472 | 0.805 | **10.5x** | 4.08e-17 | 0.00e+00 | Yes (Exact) |
| 5000 | 21.300 | 2.095 | **10.2x** | 4.16e-17 | 0.00e+00 | Yes (Exact) |

## 2. Streaming Buffer Size Trade-offs

Evaluated on a synthetic stream of 5000 samples (chunk size 100) with fixed decision boundaries.

| Buffer Size | Retrains | Total Time (s) | Streaming Pipeline Throughput (samples/s) | Mean Fit Latency / Chunk (ms) | Mean Fit Latency / Retrain Event (ms) | Mean Scoring Latency / Chunk (ms) | Estimated Structural Memory (bytes) | Prequential Acc |
|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 0.106 | 47182.8 | 2.05 | 2.05 | 0.057 | 5,136 | 0.7833 |
| 100 | 50 | 0.105 | 47637.3 | 2.03 | 2.03 | 0.056 | 5,136 | 0.7833 |
| 250 | 16 | 0.077 | 64743.5 | 1.47 | 4.52 | 0.058 | 15,248 | 0.8234 |
| 500 | 10 | 0.068 | 73053.0 | 1.30 | 6.35 | 0.055 | 23,760 | 0.8360 |
| 1000 | 5 | 0.058 | 85714.9 | 1.10 | 10.66 | 0.052 | 44,464 | 0.8405 |
| 2500 | 2 | 0.052 | 96083.5 | 0.99 | 24.07 | 0.033 | 104,592 | 0.8508 |

## 3. Analysis & Key Insights

- **Vectorisation & Partition Equivalence:** Vectorised candidate split evaluation provides a ~6x to ~10x measured speedup over naive iterative split loops across sample sizes $N \in [100, 5000]$. Both implementations produce identical sample partitions (`partition_match = True`) with gain differences bounded within machine epsilon.
- **Buffer Size vs Pipeline Throughput:** On this synthetic dataset with fixed decision boundaries, larger buffer sizes decrease the frequency of retraining cycles, resulting in higher overall streaming pipeline processing throughput (from ~45k samples/sec at buffer 50 up to ~95k samples/sec at buffer 2500).
- **Structural Memory Footprint:** The memory metric measures the estimated structural memory footprint of retained model objects and pending buffers (not full OS process resident set size). The structural footprint scales proportionally with buffer capacity (from ~5 KB to ~105 KB).
- **Prequential Integrity:** The prequential test-then-train protocol strictly evaluates incoming chunks before retraining, ensuring uncorrupted evaluation.