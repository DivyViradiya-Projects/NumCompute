"""Stream Trainer and Chunk Coordinator.

Coordinates chunk-wise training, prequential evaluation (test-then-train),
timing instrumentation, recursive memory estimation, and per-chunk structured logging.
"""
from __future__ import annotations

import copy
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

__all__ = ["StreamTrainer"]


def _estimate_tree_node_bytes(node: Any) -> int:
    """Recursively estimate memory footprint of a decision tree node structure."""
    if node is None:
        return 0
    total = sys.getsizeof(node)
    probs = getattr(node, "probabilities", None)
    if isinstance(probs, np.ndarray):
        total += probs.nbytes
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if left is not None:
        total += _estimate_tree_node_bytes(left)
    if right is not None:
        total += _estimate_tree_node_bytes(right)
    return total


def _estimate_memory(obj: Any) -> int:
    """Compute approximate structural memory footprint in bytes."""
    total_bytes = sys.getsizeof(obj)

    # Check for buffers in tree or pipeline
    for attr in ("_X_buffer", "_y_buffer", "_raw_X_buffer", "_raw_y_buffer"):
        buf = getattr(obj, attr, None)
        if isinstance(buf, list):
            total_bytes += sys.getsizeof(buf)
            for item in buf:
                if isinstance(item, np.ndarray):
                    total_bytes += item.nbytes

    # Check tree structure if standalone DecisionTreeClassifier
    tree_root = getattr(obj, "tree_", None)
    if tree_root is not None:
        total_bytes += _estimate_tree_node_bytes(tree_root)

    # Check pipeline steps and committed transformers
    for steps_attr in ("steps", "_committed_transformers"):
        steps = getattr(obj, steps_attr, None)
        if isinstance(steps, list):
            total_bytes += sys.getsizeof(steps)
            for name, step in steps:
                total_bytes += sys.getsizeof(step)
                for sub_attr in ("mean_", "var_", "scale_", "statistics_"):
                    val = getattr(step, sub_attr, None)
                    if isinstance(val, np.ndarray):
                        total_bytes += val.nbytes
                step_tree = getattr(step, "tree_", None)
                if step_tree is not None:
                    total_bytes += _estimate_tree_node_bytes(step_tree)

    return int(total_bytes)


class StreamTrainer:
    """Coordinates streaming workflow with prequential evaluation and logs.

    Implements a strict prequential protocol (test-then-train): each incoming chunk
    is evaluated against the currently fitted model *before* passing it to ``partial_fit``.
    Records per-chunk metrics, timing, retrain events, and structural memory footprint.

    Parameters
    ----------
    pipeline_or_estimator : object
        A Pipeline or standalone DecisionTreeClassifier.
    """

    def __init__(self, pipeline_or_estimator: Any) -> None:
        self.model = pipeline_or_estimator
        self.logs_: List[Dict[str, Any]] = []
        self.chunk_index_: int = 0
        self.n_samples_seen_: int = 0
        self._cumulative_correct: int = 0
        self._cumulative_evaluated: int = 0

    def score_chunk(self, X_chunk: Any, y_chunk: Any) -> Optional[float]:
        """Evaluate the current model on incoming chunk before training (pure evaluation).

        Parameters
        ----------
        X_chunk : array-like
            Incoming feature samples.
        y_chunk : array-like
            Incoming ground truth labels.

        Returns
        -------
        accuracy : float or None
            Chunk accuracy in [0.0, 1.0], or None if the model is not fitted yet.
        """
        is_fitted = getattr(self.model, "is_fitted_", False)
        if not is_fitted:
            return None

        X_arr = np.asarray(X_chunk)
        y_arr = np.asarray(y_chunk)
        if len(X_arr) == 0:
            return None

        if X_arr.ndim != 2:
            raise ValueError(f"X_chunk must be 2-D, got {X_arr.ndim}-D")
        if y_arr.ndim != 1:
            raise ValueError(f"y_chunk must be 1-D, got {y_arr.ndim}-D")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X_chunk length ({len(X_arr)}) and y_chunk length ({len(y_arr)}) must match"
            )

        y_pred = self.model.predict(X_arr)
        correct = int(np.sum(y_pred == y_arr))
        return float(correct / len(y_arr))

    def fit_chunk(self, X_chunk: Any, y_chunk: Any) -> Dict[str, Any]:
        """Update model with an incoming training chunk.

        Parameters
        ----------
        X_chunk : array-like
            Incoming feature chunk.
        y_chunk : array-like
            Incoming target labels.

        Returns
        -------
        log_entry : dict
            Metrics, timings, and retraining status for this chunk.
        """
        X_arr = np.asarray(X_chunk)
        y_arr = np.asarray(y_chunk)
        n_samples = len(X_arr)

        if n_samples == 0:
            return {}

        if X_arr.ndim != 2:
            raise ValueError(f"X_chunk must be 2-D, got {X_arr.ndim}-D")
        if y_arr.ndim != 1:
            raise ValueError(f"y_chunk must be 1-D, got {y_arr.ndim}-D")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X_chunk length ({len(X_arr)}) and y_chunk length ({len(y_arr)}) must match"
            )

        retrains_before = getattr(self.model, "n_retrains_", 0)

        # 1. Prequential evaluation (test-then-train)
        score_start = time.perf_counter()
        chunk_acc = self.score_chunk(X_arr, y_arr)
        score_time = time.perf_counter() - score_start

        if chunk_acc is not None:
            n_eval = len(y_arr)
            correct_eval = int(round(chunk_acc * n_eval))
            self._cumulative_correct += correct_eval
            self._cumulative_evaluated += n_eval

        # 2. Incremental model update
        fit_start = time.perf_counter()
        self.model.partial_fit(X_arr, y_arr)
        fit_time = time.perf_counter() - fit_start

        retrains_after = getattr(self.model, "n_retrains_", 0)
        retrained = retrains_after > retrains_before

        self.n_samples_seen_ += n_samples

        if self._cumulative_evaluated > 0:
            cum_acc: Optional[float] = float(self._cumulative_correct / self._cumulative_evaluated)
            cum_err: Optional[float] = float(1.0 - cum_acc)
        else:
            cum_acc = None
            cum_err = None

        log_entry: Dict[str, Any] = {
            "chunk_index": self.chunk_index_,
            "n_samples": n_samples,
            "cumulative_samples": self.n_samples_seen_,
            "retrained": retrained,
            "retrain_count": retrains_after,
            "fit_time": float(fit_time),
            "score_time": float(score_time),
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": cum_acc,
            "cumulative_error": cum_err,
            "n_buffered_samples": getattr(self.model, "n_buffered_samples_", 0),
            "memory_bytes": _estimate_memory(self.model),
        }

        self.logs_.append(log_entry)
        self.chunk_index_ += 1
        return copy.deepcopy(log_entry)

    def step(self, X_chunk: Any, y_chunk: Any) -> Dict[str, Any]:
        """Alias for fit_chunk."""
        return self.fit_chunk(X_chunk, y_chunk)

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return deep copy of history of structured chunk logs."""
        return copy.deepcopy(self.logs_)
