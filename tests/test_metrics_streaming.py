"""Tests for StreamingMetrics and RollingMetrics."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.metrics import RollingMetrics, StreamingMetrics


# =====================================================================
# StreamingMetrics Tests
# =====================================================================
def test_streaming_metrics_explicit_labels_init_and_rejection():
    # Explicit labels: must initialize confusion matrix immediately
    sm = StreamingMetrics(labels=["cat", "dog"])
    assert sm.confusion_matrix_ is not None
    assert sm.confusion_matrix_.shape == (2, 2)
    np.testing.assert_array_equal(sm.confusion_matrix_, np.zeros((2, 2)))

    # Valid update
    sm.update(["cat", "dog"], ["cat", "cat"])
    assert sm.n_samples_seen_ == 2
    assert sm.confusion_matrix_.sum() == 2

    # Unknown label under fixed labels must raise ValueError
    with pytest.raises(ValueError, match="unknown true label 'bird'"):
        sm.update(["bird"], ["cat"])

    with pytest.raises(ValueError, match="unknown predicted label 'fish'"):
        sm.update(["cat"], ["fish"])


def test_streaming_metrics_explicit_labels_validation():
    # Empty labels raises
    with pytest.raises(ValueError, match="must not be empty"):
        StreamingMetrics(labels=[])

    # Duplicate labels raises
    with pytest.raises(ValueError, match="must be unique"):
        StreamingMetrics(labels=["a", "b", "a"])


def test_streaming_metrics_cumulative_accuracy():
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
    # Ensure sum of confusion matrix equals n_samples_seen_
    assert sm.confusion_matrix_.sum() == sm.n_samples_seen_


def test_streaming_metrics_confusion_matrix_and_macro_scores():
    yt1 = np.array([0, 1, 2])
    yp1 = np.array([0, 1, 2])  # perfect

    yt2 = np.array([0, 1, 2])
    yp2 = np.array([1, 1, 0])  # class 0 pred as 1, class 2 pred as 0

    sm = StreamingMetrics(average="macro")
    sm.update(yt1, yp1)
    sm.update(yt2, yp2)

    cm = sm.confusion_matrix_
    assert cm is not None
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

    sm = StreamingMetrics(average="binary", pos_label="pos")
    sm.update(yt, yp)

    prec, rec, f1_val = sm.compute_precision_recall_f1()
    assert prec == 0.5
    assert rec == 0.5
    assert f1_val == 0.5


def test_streaming_metrics_zero_division():
    yt = np.array([0, 0])
    yp = np.array([1, 1])

    sm = StreamingMetrics(average="macro", zero_division=0.0)
    sm.update(yt, yp)
    res = sm.result()
    assert res["accuracy"] == 0.0


def test_streaming_metrics_auc_with_continuous_scores():
    yt1 = np.array([0, 1])
    yp1 = np.array([0, 1])
    yscore1 = np.array([0.1, 0.9])

    yt2 = np.array([0, 1])
    yp2 = np.array([0, 1])
    yscore2 = np.array([0.2, 0.8])

    sm = StreamingMetrics()
    # When no continuous score provided, AUC is None
    sm.update(yt1, yp1)
    assert sm.compute_auc() is None

    # When continuous score is provided
    sm.reset()
    sm.update(yt1, yp1, y_score_chunk=yscore1)
    sm.update(yt2, yp2, y_score_chunk=yscore2)

    auc_val = sm.compute_auc()
    assert auc_val is not None
    assert auc_val == 1.0


def test_streaming_metrics_reset():
    # Dynamic labels reset
    sm = StreamingMetrics().update([0, 1], [0, 1])
    assert sm.n_samples_seen_ == 2
    sm.reset()
    assert sm.n_samples_seen_ == 0
    assert sm.confusion_matrix_ is None
    assert sm.labels is None

    # Fixed labels reset retains structure with zeros
    sm_fixed = StreamingMetrics(labels=[0, 1]).update([0, 1], [0, 1])
    sm_fixed.reset()
    assert sm_fixed.n_samples_seen_ == 0
    assert sm_fixed.confusion_matrix_ is not None
    assert sm_fixed.confusion_matrix_.shape == (2, 2)
    np.testing.assert_array_equal(sm_fixed.confusion_matrix_, np.zeros((2, 2)))


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
    rm = RollingMetrics(window_size=3)

    rm.update([1, 1, 1], [1, 1, 1])
    assert rm.result()["accuracy"] == 1.0
    assert rm.result()["window_samples"] == 3

    # Push 3 new errors -> completely evicts the older 3 samples
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


def test_rolling_metrics_invalid_window_size():
    with pytest.raises(ValueError, match="window_size"):
        RollingMetrics(window_size=0)

    with pytest.raises(ValueError, match="window_size"):
        RollingMetrics(window_size=True)
