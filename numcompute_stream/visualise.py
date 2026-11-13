"""Reusable Matplotlib Visualisations for Streaming Evaluation.

Provides:
- plot_metric_over_time: Line plot of accuracy or error across chunks with retraining markers.
- compare_models: Side-by-side performance trajectory comparison between two model configurations.
- plot_predictions_vs_ground_truth: Discrete classification agreement and confusion visualization.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure headless-safe backend for automated test environments
if matplotlib.get_backend().lower() == "agg":
    pass

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
    """Plot metric values across chunks with optional retraining event markers.

    Parameters
    ----------
    metric_values : sequence of float
        Sequential metric values (e.g., accuracy or error).
    title : str, optional
        Chart title. Default "Metric Over Time".
    ylabel : str, optional
        Y-axis label. Default "Accuracy".
    xlabel : str, optional
        X-axis label. Default "Chunk Index".
    retrain_points : sequence of int or None, optional
        Chunk indices at which retraining occurred.
    save_path : str or None, optional
        File path to save the generated plot. Default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    vals = [v for v in metric_values if v is not None]
    if len(vals) == 0:
        raise ValueError("metric_values must not be empty or all-None")

    x = np.arange(len(metric_values))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(x, metric_values, marker="o", color="#1f77b4", linewidth=1.8, label=ylabel)

    if retrain_points:
        first = True
        for pt in retrain_points:
            if 0 <= pt < len(metric_values):
                ax.axvline(
                    x=pt,
                    color="#d62728",
                    linestyle="--",
                    alpha=0.65,
                    label="Retrain Event" if first else "",
                )
                first = False

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax


def compare_models(
    metric1: Sequence[float],
    metric2: Sequence[float],
    labels: Tuple[str, str] = ("Config A", "Config B"),
    title: str = "Model Comparison",
    ylabel: str = "Accuracy",
    xlabel: str = "Chunk Index",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare performance trajectories between two streaming configurations.

    Parameters
    ----------
    metric1 : sequence of float
        Metric trajectory of configuration 1.
    metric2 : sequence of float
        Metric trajectory of configuration 2.
    labels : (str, str), optional
        Legend labels for the two configurations. Default ("Config A", "Config B").
    title : str, optional
        Chart title. Default "Model Comparison".
    ylabel : str, optional
        Y-axis label. Default "Accuracy".
    xlabel : str, optional
        X-axis label. Default "Chunk Index".
    save_path : str or None, optional
        File path to save figure. Default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if len(metric1) == 0 or len(metric2) == 0:
        raise ValueError("metric sequences must not be empty")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x1 = np.arange(len(metric1))
    x2 = np.arange(len(metric2))

    ax.plot(x1, metric1, marker="s", color="#1f77b4", linewidth=1.8, label=labels[0])
    ax.plot(x2, metric2, marker="^", color="#ff7f0e", linewidth=1.8, label=labels[1])

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    title: str = "Predictions vs Ground Truth",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot sample-by-sample predictions compared against ground truth.

    Parameters
    ----------
    y_true : sequence
        Ground truth labels.
    y_pred : sequence
        Predicted labels.
    title : str, optional
        Chart title. Default "Predictions vs Ground Truth".
    save_path : str or None, optional
        File path to save figure. Default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if len(yt) != len(yp):
        raise ValueError(f"Lengths must match: len(y_true)={len(yt)}, len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Input sequences must not be empty")

    n_samples = len(yt)
    x = np.arange(n_samples)
    correct = yt == yp

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(x[correct], yt[correct], color="#2ca02c", marker="o", s=35, label="Correct Prediction")
    if np.any(~correct):
        ax.scatter(
            x[~correct],
            yt[~correct],
            color="#d62728",
            marker="x",
            s=50,
            label="Misclassified (True)",
        )
        ax.scatter(
            x[~correct],
            yp[~correct],
            color="#ff7f0e",
            marker="^",
            s=40,
            label="Misclassified (Predicted)",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Class Label", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax
