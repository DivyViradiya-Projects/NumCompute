"""Tests for Matplotlib streaming visualisations."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_plot_metric_over_time(tmp_path):
    metric_values = [0.6, 0.7, 0.65, 0.8, 0.85]
    retrain_points = [1, 3]

    fig, ax = plot_metric_over_time(
        metric_values,
        title="Accuracy Over Time",
        ylabel="Accuracy",
        retrain_points=retrain_points,
    )
    assert fig is not None
    assert ax is not None
    assert ax.get_ylabel() == "Accuracy"
    assert ax.get_title() == "Accuracy Over Time"

    # Save to file test
    out_file = tmp_path / "metric.png"
    fig, _ = plot_metric_over_time(metric_values, save_path=str(out_file))
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_compare_models(tmp_path):
    m1 = [0.5, 0.6, 0.7, 0.75]
    m2 = [0.45, 0.65, 0.72, 0.82]

    fig, ax = compare_models(
        m1,
        m2,
        labels=("Buffer=64", "Buffer=256"),
        title="Buffer Comparison",
    )
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Buffer Comparison"

    out_file = tmp_path / "compare.png"
    fig, _ = compare_models(m1, m2, save_path=str(out_file))
    assert out_file.is_file()


def test_plot_predictions_vs_ground_truth(tmp_path):
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]  # two misclassifications

    fig, ax = plot_predictions_vs_ground_truth(
        y_true,
        y_pred,
        title="Ground Truth vs Predictions",
    )
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Ground Truth vs Predictions"

    out_file = tmp_path / "preds.png"
    fig, _ = plot_predictions_vs_ground_truth(y_true, y_pred, save_path=str(out_file))
    assert out_file.is_file()


def test_visualise_validation_errors():
    with pytest.raises(ValueError, match="empty"):
        plot_metric_over_time([])

    with pytest.raises(ValueError, match="empty"):
        compare_models([], [1.0])

    with pytest.raises(ValueError, match="Lengths must match"):
        plot_predictions_vs_ground_truth([0, 1], [0])
