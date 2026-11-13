"""Incremental Statistics.

Provides StreamingStats and update_stats for running mean, variance (Welford/Chan),
quantiles, and histograms over data streams.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

__all__ = ["StreamingStats", "update_stats"]


class StreamingStats:
    """Online per-feature statistics tracker (Welford / Chan)."""

    def __init__(self, ddof: int = 0) -> None:
        self.ddof = ddof
        self.n_features_: Optional[int] = None
        self._count = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)

    def update(self, x) -> "StreamingStats":
        """Add a batch or chunk of observations."""
        raise NotImplementedError("Will be implemented in Phase 3")

    @property
    def count(self) -> np.ndarray:
        """Count of observations per feature."""
        return self._count.copy()

    @property
    def mean(self) -> np.ndarray:
        """Running mean per feature."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def var(self, ddof: Optional[int] = None) -> np.ndarray:
        """Running variance per feature."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def std(self, ddof: Optional[int] = None) -> np.ndarray:
        """Running standard deviation per feature."""
        return np.sqrt(self.var(ddof))

    def quantile(self, q: float) -> np.ndarray:
        """Estimate quantile q in [0, 1]."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def histogram(self, bins: int = 10, range: Optional[Tuple[float, float]] = None):
        """Streaming histogram counts and bin edges."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def reset(self) -> "StreamingStats":
        """Reset internal accumulator state."""
        self.n_features_ = None
        self._count = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)
        return self


def update_stats(existing_stats: Optional[StreamingStats], X_chunk) -> StreamingStats:
    """Convenience functional update for streaming statistics."""
    if existing_stats is None:
        stats = StreamingStats()
    else:
        stats = existing_stats
    return stats.update(X_chunk)

