"""Streaming Pipeline.

Chains transformers with a final estimator, orchestrating incremental updates
and managing a raw-feature buffer to ensure consistent feature coordinates upon retraining.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from numcompute.utils import NotFittedError

__all__ = ["Pipeline"]


class Pipeline:
    """Streaming Pipeline with raw-pending buffer management.

    Parameters
    ----------
    steps : sequence of (name, estimator_or_transformer)
        List of (name, step) tuples where all steps but the last are
        transformers and the final step is an estimator.
    """

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        self.steps = list(steps)
        self._validate_steps()
        self._raw_X_buffer: list[np.ndarray] = []
        self._raw_y_buffer: list[np.ndarray] = []
        self._pending_count: int = 0
        self.is_fitted_: bool = False

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("steps must not be empty")
        names = [s[0] for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}")

    @property
    def named_steps(self) -> dict[str, Any]:
        """Access pipeline steps by name."""
        return dict(self.steps)

    def fit(self, X, y) -> "Pipeline":
        """Fit all transformers and the final estimator on batch (X, y)."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def partial_fit(self, X_chunk, y_chunk) -> "Pipeline":
        """Process an incoming chunk with raw buffer consistency."""
        raise NotImplementedError("Will be implemented in Phase 4")

    def predict(self, X) -> np.ndarray:
        """Transform X through fitted transformers and predict with final estimator."""
        if not self.is_fitted_:
            raise NotFittedError("Pipeline is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 4")

    def score(self, X, y) -> float:
        """Score predictions against true labels y."""
        raise NotImplementedError("Will be implemented in Phase 4")

