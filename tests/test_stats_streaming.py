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


def test_update_stats_functional():
    c1 = [[1.0], [2.0]]
    c2 = [[3.0], [4.0]]

    s = update_stats(None, c1)
    assert s.count[0] == 2.0
    s = update_stats(s, c2)
    assert s.count[0] == 4.0
    assert s.mean[0] == 2.5
