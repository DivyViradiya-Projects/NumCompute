"""Stateful Streaming Evaluation Metrics.

Provides StreamingMetrics and RollingMetrics for tracking classification metrics
(accuracy, precision, recall, F1, confusion matrix, AUC) across sequential chunks.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1,
    roc_auc_score,
)

__all__ = [
    "StreamingMetrics",
    "RollingMetrics",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "roc_auc_score",
]


class StreamingMetrics:
    """Cumulative classification metrics accumulator."""

    def __init__(self, labels: Optional[Sequence] = None) -> None:
        self.labels = list(labels) if labels is not None else None
        self.confusion_matrix_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self._all_y_true: list[np.ndarray] = []
        self._all_y_pred: list[np.ndarray] = []

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingMetrics":
        """Accumulate predictions and ground truth from a chunk."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def result(self) -> Dict[str, float]:
        """Compute cumulative metrics dictionary."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def reset(self) -> "StreamingMetrics":
        """Reset internal accumulator state."""
        self.confusion_matrix_ = None
        self.n_samples_seen_ = 0
        self._all_y_true.clear()
        self._all_y_pred.clear()
        return self


class RollingMetrics:
    """Rolling window metrics evaluator over the most recent W samples."""

    def __init__(self, window_size: int = 500) -> None:
        self.window_size = window_size
        self._y_true_window: list = []
        self._y_pred_window: list = []

    def update(self, y_true_chunk, y_pred_chunk) -> "RollingMetrics":
        """Update sliding window with incoming chunk."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def result(self) -> Dict[str, float]:
        """Compute metrics over current window."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def reset(self) -> "RollingMetrics":
        """Clear the window."""
        self._y_true_window.clear()
        self._y_pred_window.clear()
        return self

