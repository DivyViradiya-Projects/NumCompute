"""Tests for StreamingStats and update_stats."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats, update_stats


def test_streaming_stats_mean_and_var_match_numpy():
    c1 = np.array([[1.0, 10.0], [2.0, 20.0]])
    c2 = np.array([[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    c3 = np.array([[6.0, 60.0]])
    full = np.vstack([c1, c2, c3])

    stats = StreamingStats()
    stats.update(c1)
    stats.update(c2)
    stats.update(c3)

    assert stats.n_features_ == 2
    np.testing.assert_array_equal(stats.count, [6.0, 6.0])

    np.testing.assert_allclose(stats.mean, np.mean(full, axis=0))
    np.testing.assert_allclose(stats.var(ddof=0), np.var(full, axis=0, ddof=0))
    np.testing.assert_allclose(stats.std(ddof=0), np.std(full, axis=0, ddof=0))

    np.testing.assert_allclose(stats.var(ddof=1), np.var(full, axis=0, ddof=1))
    np.testing.assert_allclose(stats.std(ddof=1), np.std(full, axis=0, ddof=1))


def test_streaming_stats_large_offset_stability():
    base = 1e9
    chunk1 = np.array([[base + 1.0], [base + 2.0]])
    chunk2 = np.array([[base + 3.0], [base + 4.0]])
    full = np.vstack([chunk1, chunk2])

    stats = StreamingStats().update(chunk1).update(chunk2)

    np.testing.assert_allclose(stats.mean, np.mean(full, axis=0))
    np.testing.assert_allclose(stats.var(ddof=1), [5.0 / 3.0], atol=1e-8)


def test_streaming_stats_skips_nan():
    c1 = np.array([[1.0, np.nan], [2.0, 20.0]])
    c2 = np.array([[np.nan, 30.0], [4.0, 40.0]])

    stats = StreamingStats().update(c1).update(c2)

    np.testing.assert_array_equal(stats.count, [3.0, 3.0])
    np.testing.assert_allclose(stats.mean, [7.0 / 3.0, 30.0])


def test_streaming_stats_quantiles_scalar_and_sequence():
    data = np.arange(1, 101, dtype=float).reshape(50, 2)
    c1 = data[:25]
    c2 = data[25:]

    stats = StreamingStats().update(c1).update(c2)

    # Scalar q -> 1D output shape (n_features,)
    median = stats.quantile(0.5)
    assert median.shape == (2,)
    np.testing.assert_allclose(median, np.median(data, axis=0))

    # Sequence q -> 2D output shape (len(q), n_features)
    q_vals = [0.25, 0.75]
    res = stats.quantile(q_vals)
    assert res.shape == (2, 2)
    np.testing.assert_allclose(res, np.quantile(data, q=q_vals, axis=0, method="linear"))


def test_streaming_stats_incremental_histogram():
    # Predefined fixed bin histogram
    stats = StreamingStats(hist_bins=5, hist_range=(0.0, 10.0))
    c1 = np.array([[1.0], [2.0], [3.0]])
    c2 = np.array([[6.0], [7.0], [8.0]])

    stats.update(c1)
    stats.update(c2)

    counts, edges = stats.histogram()
    expected_counts, expected_edges = np.histogram(
        np.vstack([c1, c2]), bins=5, range=(0.0, 10.0)
    )

    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(edges, expected_edges)


def test_streaming_stats_reset():
    stats = StreamingStats().update([[1.0, 2.0], [3.0, 4.0]])
    assert stats.count[0] == 2.0

    stats.reset()
    assert stats.n_features_ is None
    assert len(stats.count) == 0


def test_streaming_stats_errors():
    stats = StreamingStats()
    with pytest.raises(ValueError, match="No data has been observed yet"):
        stats.quantile(0.5)

    with pytest.raises(ValueError, match="No data has been observed yet"):
        stats.histogram()

    stats.update([[1.0, 2.0]])
    with pytest.raises(ValueError, match="initialized with 2 features"):
        stats.update([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="ddof must be a non-negative int"):
        stats.var(ddof=-1)

    with pytest.raises(ValueError, match="ddof must be a non-negative int"):
        stats.var(ddof=True)

    with pytest.raises(ValueError, match="Quantile q must be in"):
        stats.quantile(-0.1)

    with pytest.raises(ValueError, match="Quantile values must be finite"):
        stats.quantile(np.nan)


def test_streaming_stats_hist_configuration_validation():
    # Negative / zero bins
    with pytest.raises(ValueError, match="hist_bins integer count must be positive"):
        StreamingStats(hist_bins=0)

    with pytest.raises(ValueError, match="hist_bins integer count must be positive"):
        StreamingStats(hist_bins=-5)

    # Boolean bins rejected
    with pytest.raises(TypeError, match="hist_bins cannot be a boolean"):
        StreamingStats(hist_bins=True)

    # Invalid range: min >= max
    with pytest.raises(ValueError, match="lower boundary must be strictly less than upper boundary"):
        StreamingStats(hist_bins=10, hist_range=(10.0, 5.0))

    with pytest.raises(ValueError, match="lower boundary must be strictly less than upper boundary"):
        StreamingStats(hist_bins=10, hist_range=(5.0, 5.0))

    # Infinite range
    with pytest.raises(ValueError, match="hist_range values must be finite"):
        StreamingStats(hist_bins=10, hist_range=(0.0, np.inf))

    # Explicit edges: non-monotonic or duplicates
    with pytest.raises(ValueError, match="strictly increasing"):
        StreamingStats(hist_bins=[0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        StreamingStats(hist_bins=[0.0, 1.0, 1.0, 2.0])

    # Explicit edges: NaN or Inf
    with pytest.raises(ValueError, match="finite numeric values"):
        StreamingStats(hist_bins=[0.0, np.nan, 2.0])

    # Explicit edges: length < 2
    with pytest.raises(ValueError, match="length >= 2"):
        StreamingStats(hist_bins=[1.0])


def test_streaming_stats_hist_bins_without_range_initializes_from_first_chunk():
    stats = StreamingStats(hist_bins=4)
    c1 = np.array([[0.0], [4.0]])
    c2 = np.array([[1.0], [3.0]])

    stats.update(c1)
    # Range should be inferred as [0.0, 4.0], with 4 bins: [0, 1], [1, 2], [2, 3], [3, 4]
    counts1, edges1 = stats.histogram()
    assert len(counts1) == 4
    np.testing.assert_allclose(edges1, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(counts1, [1, 0, 0, 1])

    # Subsequent chunk accumulates into the same established bins
    stats.update(c2)
    counts2, edges2 = stats.histogram()
    np.testing.assert_allclose(edges2, edges1)
    # NumPy bins [0, 1), [1, 2), [2, 3), [3, 4]: values 0->b0, 1->b1, 3->b3, 4->b3
    np.testing.assert_array_equal(counts2, [1, 1, 0, 2])


def test_streaming_stats_histogram_out_of_range_handling():
    # Fixed range [0.0, 10.0]
    stats = StreamingStats(hist_bins=2, hist_range=(0.0, 10.0))
    # Bins are [0, 5) and [5, 10]
    # -5 is below range, 15 is above range, 2 is in bin 0, 7 is in bin 1
    stats.update([[-5.0], [2.0], [7.0], [15.0]])

    counts, edges = stats.histogram()
    np.testing.assert_array_equal(counts, [1, 1])
    np.testing.assert_allclose(edges, [0.0, 5.0, 10.0])


def test_streaming_stats_reset_with_predefined_edges():
    stats = StreamingStats(hist_bins=2, hist_range=(0.0, 10.0))
    stats.update([[2.0], [7.0]])
    counts, _ = stats.histogram()
    np.testing.assert_array_equal(counts, [1, 1])

    stats.reset()
    assert stats.n_features_ is None
    # Resetting retains the configured bin edges but zeroes counts
    counts_after, edges_after = stats.histogram()
    np.testing.assert_array_equal(counts_after, [0, 0])
    np.testing.assert_allclose(edges_after, [0.0, 5.0, 10.0])


def test_update_stats_functional():
    c1 = [[1.0], [2.0]]
    c2 = [[3.0], [4.0]]

    s = update_stats(None, c1)
    assert s.count[0] == 2.0
    s = update_stats(s, c2)
    assert s.count[0] == 4.0
    assert s.mean[0] == 2.5
