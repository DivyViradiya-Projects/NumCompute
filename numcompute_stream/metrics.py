"""Stateful Streaming Evaluation Metrics.

Provides StreamingMetrics and RollingMetrics for accumulating and evaluating
classification performance (accuracy, precision, recall, F1, confusion matrix, AUC)
across sequential data chunks.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
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


def _safe_div(numerator: float, denominator: float, zero_division: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0 else float(zero_division)


class StreamingMetrics:
    """Cumulative classification metrics accumulator across sequential chunks.

    Maintains a running confusion matrix and sample counters so metrics (accuracy,
    precision, recall, F1) are derived from total accumulated counts rather than
    erroneously averaging chunk-level percentages. Also records true labels and
    prediction scores to compute cumulative ROC AUC.

    Parameters
    ----------
    labels : sequence or None, optional
        Set of class labels. If None, classes are inferred and ordered
        deterministically from observed data chunks.
    average : {"binary", "macro", "micro"}, optional
        Averaging strategy for precision, recall, and F1. Default "macro".
    pos_label : int, str, or bool, optional
        Class to report when average="binary". Default 1.
    zero_division : float, optional
        Value to return when denominator is zero. Default 0.0.
    """

    def __init__(
        self,
        labels: Optional[Sequence[Any]] = None,
        average: str = "macro",
        pos_label: Any = 1,
        zero_division: float = 0.0,
    ) -> None:
        if average not in ("binary", "macro", "micro"):
            raise ValueError(
                f"average must be 'binary', 'macro', or 'micro', got {average!r}"
            )
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)

        self.labels: Optional[List[Any]] = list(labels) if labels is not None else None
        self._labels_fixed = labels is not None

        # Cumulative confusion matrix: shape (n_classes, n_classes)
        self.confusion_matrix_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.n_chunks_seen_: int = 0

        # Accumulated true labels and predictions for AUC
        self._all_y_true: List[np.ndarray] = []
        self._all_y_pred: List[np.ndarray] = []

    def _sync_labels(self, yt: np.ndarray, yp: np.ndarray) -> None:
        unique_seen = np.unique(np.concatenate([yt, yp]))
        if self.labels is None:
            self.labels = sorted(unique_seen.tolist())
            n = len(self.labels)
            self.confusion_matrix_ = np.zeros((n, n), dtype=np.int64)
        elif not self._labels_fixed:
            # Check for new unseen labels
            new_labels = [l for l in unique_seen if l not in self.labels]
            if new_labels:
                old_labels = list(self.labels)
                self.labels = sorted(old_labels + new_labels)
                old_cm = self.confusion_matrix_
                assert old_cm is not None
                n_new = len(self.labels)
                new_cm = np.zeros((n_new, n_new), dtype=np.int64)
                # Remap existing counts
                old_idx = [self.labels.index(l) for l in old_labels]
                for i_old, i_new in enumerate(old_idx):
                    for j_old, j_new in enumerate(old_idx):
                        new_cm[i_new, j_new] = old_cm[i_old, j_old]
                self.confusion_matrix_ = new_cm

    def update(self, y_true_chunk: Any, y_pred_chunk: Any) -> "StreamingMetrics":
        """Accumulate predictions and ground truth labels from a chunk.

        Parameters
        ----------
        y_true_chunk : array-like of shape (n_samples,)
            True class labels.
        y_pred_chunk : array-like of shape (n_samples,)
            Predicted labels or scores.

        Returns
        -------
        self : StreamingMetrics
        """
        yt = np.asarray(y_true_chunk)
        yp = np.asarray(y_pred_chunk)

        if yt.ndim != 1 or yp.ndim != 1:
            raise ValueError(
                f"y_true and y_pred must be 1-D arrays, got {yt.shape} and {yp.shape}"
            )
        if len(yt) != len(yp):
            raise ValueError(
                f"y_true length ({len(yt)}) and y_pred length ({len(yp)}) must match"
            )
        if len(yt) == 0:
            return self

        self._sync_labels(yt, yp)
        assert self.labels is not None
        assert self.confusion_matrix_ is not None

        # Build chunk confusion matrix using label index mapping
        label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        for t, p in zip(yt, yp):
            if t in label_to_idx and p in label_to_idx:
                self.confusion_matrix_[label_to_idx[t], label_to_idx[p]] += 1

        self.n_samples_seen_ += len(yt)
        self.n_chunks_seen_ += 1
        self._all_y_true.append(yt.copy())
        self._all_y_pred.append(yp.copy())
        return self

    def compute_accuracy(self) -> float:
        """Calculate cumulative accuracy."""
        if self.confusion_matrix_ is None or self.n_samples_seen_ == 0:
            return 0.0
        correct = float(np.trace(self.confusion_matrix_))
        return _safe_div(correct, self.n_samples_seen_, self.zero_division)

    def compute_precision_recall_f1(self) -> Tuple[float, float, float]:
        """Compute cumulative precision, recall, and F1 based on configured averaging."""
        if self.confusion_matrix_ is None or self.labels is None or self.n_samples_seen_ == 0:
            return self.zero_division, self.zero_division, self.zero_division

        cm = self.confusion_matrix_
        n_classes = len(self.labels)

        if self.average == "micro":
            tp_total = float(np.trace(cm))
            total = float(np.sum(cm))
            prec = _safe_div(tp_total, total, self.zero_division)
            rec = _safe_div(tp_total, total, self.zero_division)
            f1_score = _safe_div(2 * prec * rec, prec + rec, self.zero_division)
            return prec, rec, f1_score

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp

        prec_per_class = np.array([
            _safe_div(tp[i], tp[i] + fp[i], self.zero_division) for i in range(n_classes)
        ])
        rec_per_class = np.array([
            _safe_div(tp[i], tp[i] + fn[i], self.zero_division) for i in range(n_classes)
        ])
        f1_per_class = np.array([
            _safe_div(
                2 * prec_per_class[i] * rec_per_class[i],
                prec_per_class[i] + rec_per_class[i],
                self.zero_division,
            )
            for i in range(n_classes)
        ])

        if self.average == "binary":
            if self.pos_label not in self.labels:
                return self.zero_division, self.zero_division, self.zero_division
            idx = self.labels.index(self.pos_label)
            return float(prec_per_class[idx]), float(rec_per_class[idx]), float(f1_per_class[idx])
        elif self.average == "macro":
            return (
                float(np.mean(prec_per_class)),
                float(np.mean(rec_per_class)),
                float(np.mean(f1_per_class)),
            )

        return self.zero_division, self.zero_division, self.zero_division

    def compute_auc(self) -> Optional[float]:
        """Compute cumulative ROC AUC if binary targets are observed."""
        if not self._all_y_true:
            return None
        y_true_all = np.concatenate(self._all_y_true)
        y_pred_all = np.concatenate(self._all_y_pred)

        unique_true = np.unique(y_true_all)
        if len(unique_true) != 2:
            return None

        # Binary ROC AUC score
        try:
            return float(roc_auc_score(y_true_all, y_pred_all))
        except Exception:
            return None

    def result(self) -> Dict[str, Any]:
        """Compute complete cumulative metrics summary dictionary."""
        acc = self.compute_accuracy()
        prec, rec, f1_val = self.compute_precision_recall_f1()
        res: Dict[str, Any] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1_val,
            "n_samples": self.n_samples_seen_,
            "n_chunks": self.n_chunks_seen_,
            "confusion_matrix": self.confusion_matrix_.copy() if self.confusion_matrix_ is not None else None,
        }
        auc_val = self.compute_auc()
        if auc_val is not None:
            res["auc"] = auc_val
        return res

    def reset(self) -> "StreamingMetrics":
        """Reset internal accumulator state."""
        self.confusion_matrix_ = None
        self.n_samples_seen_ = 0
        self.n_chunks_seen_ = 0
        self._all_y_true.clear()
        self._all_y_pred.clear()
        if not self._labels_fixed:
            self.labels = None
        return self


class RollingMetrics:
    """Rolling-window metrics evaluator over the most recent W samples.

    Parameters
    ----------
    window_size : int, optional
        Maximum number of samples held in the sliding evaluation window. Default 500.
    zero_division : float, optional
        Value to return when denominator is zero. Default 0.0.
    """

    def __init__(self, window_size: int = 500, zero_division: float = 0.0) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.zero_division = float(zero_division)

        self._y_true_window: deque = deque(maxlen=window_size)
        self._y_pred_window: deque = deque(maxlen=window_size)

    def update(self, y_true_chunk: Any, y_pred_chunk: Any) -> "RollingMetrics":
        """Push a chunk of observations into the rolling window.

        Parameters
        ----------
        y_true_chunk : array-like
            Ground truth labels.
        y_pred_chunk : array-like
            Predicted labels.

        Returns
        -------
        self : RollingMetrics
        """
        yt = np.asarray(y_true_chunk)
        yp = np.asarray(y_pred_chunk)

        if yt.ndim != 1 or yp.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays")
        if len(yt) != len(yp):
            raise ValueError(f"Length mismatch: {len(yt)} vs {len(yp)}")

        for t, p in zip(yt, yp):
            self._y_true_window.append(t)
            self._y_pred_window.append(p)
        return self

    def result(self) -> Dict[str, Any]:
        """Compute evaluation metrics over current sliding window."""
        n_samples = len(self._y_true_window)
        if n_samples == 0:
            return {
                "accuracy": self.zero_division,
                "precision": self.zero_division,
                "recall": self.zero_division,
                "f1": self.zero_division,
                "window_samples": 0,
            }

        yt = np.array(self._y_true_window)
        yp = np.array(self._y_pred_window)

        acc = float(np.mean(yt == yp))
        prec = float(precision(yt, yp, average="macro", zero_division=self.zero_division))
        rec = float(recall(yt, yp, average="macro", zero_division=self.zero_division))
        f1_val = float(f1(yt, yp, average="macro", zero_division=self.zero_division))

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1_val,
            "window_samples": n_samples,
        }

    def reset(self) -> "RollingMetrics":
        """Clear the rolling window."""
        self._y_true_window.clear()
        self._y_pred_window.clear()
        return self
