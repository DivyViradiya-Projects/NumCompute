"""Reusable Matplotlib Visualisations for Streaming Evaluation.

Provides:
- plot_metric_over_time
- compare_models
- plot_predictions_vs_ground_truth
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]


def plot_metric_over_time(
    metric_values: Sequence[float],
    title: str = "Metric Over Time",
    ylabel: str = "Accuracy",
    xlabel: str = "Chunk Index",
    retrain_points: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot metric values across chunks with optional retraining markers."""
    raise NotImplementedError("Will be implemented in Phase 5")


def compare_models(
    metric1: Sequence[float],
    metric2: Sequence[float],
    labels: Tuple[str, str] = ("Config A", "Config B"),
    title: str = "Model Comparison",
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare performance trajectories between two streaming configurations."""
    raise NotImplementedError("Will be implemented in Phase 5")


def plot_predictions_vs_ground_truth(
    y_true: Sequence,
    y_pred: Sequence,
    title: str = "Predictions vs Ground Truth",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot sample-by-sample predictions compared against ground truth."""
    raise NotImplementedError("Will be implemented in Phase 5")
