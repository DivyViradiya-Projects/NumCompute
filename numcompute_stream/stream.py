"""Stream Trainer and Chunk Coordinator.

Coordinates chunk-wise training, prequential evaluation, and detailed logging.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

__all__ = ["StreamTrainer"]


class StreamTrainer:
    """Coordinates streaming workflow with prequential evaluation and logs.

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

    def fit_chunk(self, X_chunk, y_chunk) -> Dict[str, Any]:
        """Update model with an incoming training chunk."""
        raise NotImplementedError("Will be implemented in Phase 5")

    def score_chunk(self, X_chunk, y_chunk) -> float:
        """Evaluate current model on incoming chunk before training (prequential)."""
        raise NotImplementedError("Will be implemented in Phase 5")

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return history of chunk logs."""
        return list(self.logs_)
