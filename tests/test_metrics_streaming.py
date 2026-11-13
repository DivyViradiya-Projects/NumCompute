"""Tests for StreamingMetrics and RollingMetrics."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.metrics import RollingMetrics, StreamingMetrics


# =====================================================================
# StreamingMetrics Tests
# =====================================================================
def test_streaming_metrics_cumulative_accuracy():
    # Chunk 1: 3/4 correct (75%)
    # Chunk 2: 1/2 correct (50%)
    # Overall: 4/6 correct = 66.67%
    yt1 = np.array([0, 1, 0, 1])
    yp1 = np.array([0, 1, 1, 1])  # error at index 2

    yt2 = np.array([0, 1])
    yp2 = np.array([1, 1])  # error at index 0

    sm = StreamingMetrics()
    sm.update(yt1, yp1)
    assert sm.compute_accuracy() == 0.75

    sm.update(yt2, yp2)
    assert np.isclose(sm.compute_accuracy(), 4.0 / 6.0)
    assert sm.n_samples_seen_ == 6
    assert sm.n_chunks_seen_ == 2


def test_streaming_metrics_confusion_matrix_and_macro_scores():
    # 3 classes: 0, 1, 2
    yt1 = np.array([0, 1, 2])
    yp1 = np.array([0, 1, 2])  # perfect

    yt2 = np.array([0, 1, 2])
    yp2 = np.array([1, 1, 0])  # class 0 pred as 1, class 2 pred as 0

    sm = StreamingMetrics(average="macro")
    sm.update(yt1, yp1)
    sm.update(yt2, yp2)

    cm = sm.confusion_matrix_
    assert cm is not None
    # True 0: one pred 0, one pred 1 -> [1, 1, 0]
    # True 1: two pred 1 -> [0, 2, 0]
    # True 2: one pred 0, one pred 2 -> [1, 0, 1]
    expected_cm = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ])
    np.testing.assert_array_equal(cm, expected_cm)

    res = sm.result()
    assert np.isclose(res["accuracy"], 4.0 / 6.0)
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0


def test_streaming_metrics_binary_pos_label():
    yt = np.array(["neg", "pos", "pos", "neg"])
    yp = np.array(["neg", "neg", "pos", "pos"])

    # For pos: TP=1, FP=1, FN=1 -> prec = 1/2, rec = 1/2, f1 = 1/2
    sm = StreamingMetrics(average="binary", pos_label="pos")
    sm.update(yt, yp)

    prec, rec, f1_val = sm.compute_precision_recall_f1()
    assert prec == 0.5
    assert rec == 0.5
    assert f1_val == 0.5


def test_streaming_metrics_zero_division():
    # All false positives for class 1
    yt = np.array([0, 0])
    yp = np.array([1, 1])

    sm = StreamingMetrics(average="macro", zero_division=0.0)
    sm.update(yt, yp)
    res = sm.result()
    assert res["accuracy"] == 0.0


def test_streaming_metrics_auc():
    yt1 = np.array([0, 1])
    yp1 = np.array([0.1, 0.9])
    yt2 = np.array([0, 1])
    yp2 = np.array([0.2, 0.8])

    sm = StreamingMetrics()
    sm.update(yt1, yp1)
    sm.update(yt2, yp2)

    auc_val = sm.compute_auc()
    assert auc_val is not None
    assert auc_val == 1.0


def test_streaming_metrics_reset():
    sm = StreamingMetrics().update([0, 1], [0, 1])
    assert sm.n_samples_seen_ == 2
    sm.reset()
    assert sm.n_samples_seen_ == 0
    assert sm.confusion_matrix_ is None


def test_streaming_metrics_validation_errors():
    with pytest.raises(ValueError, match="average"):
        StreamingMetrics(average="unsupported")

    sm = StreamingMetrics()
    with pytest.raises(ValueError, match="1-D arrays"):
        sm.update([[0]], [[0]])

    with pytest.raises(ValueError, match="match"):
        sm.update([0, 1], [0])


# =====================================================================
# RollingMetrics Tests
# =====================================================================
def test_rolling_metrics_sliding_window_eviction():
    # Window size 3
    rm = RollingMetrics(window_size=3)

    # Push 3 samples: [1, 1, 1] pred [1, 1, 1] (100% acc)
    rm.update([1, 1, 1], [1, 1, 1])
    assert rm.result()["accuracy"] == 1.0
    assert rm.result()["window_samples"] == 3

    # Push 3 new errors: [0, 0, 0] pred [1, 1, 1]
    # Completely evicts the older 3 samples! Accuracy should become 0.0
    rm.update([0, 0, 0], [1, 1, 1])
    assert rm.result()["accuracy"] == 0.0
    assert rm.result()["window_samples"] == 3


def test_rolling_metrics_partial_window():
    rm = RollingMetrics(window_size=10)
    rm.update([0, 1], [0, 1])
    res = rm.result()
    assert res["accuracy"] == 1.0
    assert res["window_samples"] == 2


def test_rolling_metrics_reset():
    rm = RollingMetrics(window_size=5).update([1, 0], [1, 0])
    rm.reset()
    assert rm.result()["window_samples"] == 0
    assert rm.result()["accuracy"] == 0.0

