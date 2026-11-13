"""Incremental Statistics.

Provides StreamingStats and update_stats for running mean, variance (Welford/Chan),
quantiles, and histograms over data streams.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from numcompute.utils import as_float_array

__all__ = ["StreamingStats", "update_stats"]


class StreamingStats:
    """Online per-feature statistics tracker using Chan et al.'s parallel formula.

    Maintains running count, mean, and M2 accumulator in O(d) memory, matching
    Welford's algorithm on single-sample updates and parallel Chan algorithm on chunk updates.
    Also tracks streaming quantiles and fixed-bin histograms.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom for default variance divisor (count - ddof). Default 0.

    Attributes
    ----------
    n_features_ : int or None
        Number of features seen.
    count : ndarray of shape (n_features,)
        Count of non-NaN observations per feature.
    mean : ndarray of shape (n_features,)
        Running arithmetic mean per feature (NaN if no observations).
    """

    def __init__(self, ddof: int = 0) -> None:
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        self.ddof = ddof

        self.n_features_: Optional[int] = None
        self._n = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)

        # Buffer for exact quantile calculation
        self._sample_buffer: List[np.ndarray] = []

        # Histogram state
        self._bin_edges: Optional[np.ndarray] = None
        self._hist_counts: Optional[np.ndarray] = None

    def _init(self, n_features: int) -> None:
        self.n_features_ = n_features
        self._n = np.zeros(n_features, dtype=float)
        self._mean = np.zeros(n_features, dtype=float)
        self._m2 = np.zeros(n_features, dtype=float)

    def update(self, x: Any) -> "StreamingStats":
        """Add a chunk or batch of observations.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features) or (n_samples,)
            Incoming chunk. A 1-D input is treated as a single feature.

        Returns
        -------
        self : StreamingStats
        """
        arr = as_float_array(x, name="x", ndim=(1, 2))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_samples, n_features = arr.shape
        if self.n_features_ is None:
            self._init(n_features)
        elif n_features != self.n_features_:
            raise ValueError(
                f"x has {n_features} features, but StreamingStats was initialized "
                f"with {self.n_features_} features"
            )

        if n_samples == 0:
            return self

        # Non-NaN mask
        valid = ~np.isnan(arr)
        chunk_n = valid.sum(axis=0).astype(float)
        filled = np.where(valid, arr, 0.0)
        chunk_mean = filled.sum(axis=0) / np.maximum(chunk_n, 1.0)
        chunk_m2 = (((filled - chunk_mean) ** 2) * valid).sum(axis=0)

        # Merge with Chan et al.'s formula
        total_n = self._n + chunk_n
        delta = chunk_mean - self._mean
        frac = chunk_n / np.maximum(total_n, 1.0)

        self._mean = self._mean + delta * frac
        self._m2 = self._m2 + chunk_m2 + (delta**2) * self._n * frac
        self._n = total_n

        # Retain for exact quantile tracking
        self._sample_buffer.append(arr.copy())

        # Update histogram counts if edges already established
        if self._bin_edges is not None and self._hist_counts is not None:
            # Flatten 2D chunk values omitting NaN
            flat_vals = arr[~np.isnan(arr)]
            if flat_vals.size > 0:
                counts, _ = np.histogram(flat_vals, bins=self._bin_edges)
                self._hist_counts += counts

        return self

    @property
    def count(self) -> np.ndarray:
        """Non-NaN observation count per feature."""
        return self._n.copy()

    @property
    def mean(self) -> np.ndarray:
        """Running arithmetic mean per feature."""
        if self.n_features_ is None:
            return np.zeros(0, dtype=float)
        return np.where(self._n > 0, self._mean, np.nan)

    def var(self, ddof: Optional[int] = None) -> np.ndarray:
        """Running variance per feature.

        Parameters
        ----------
        ddof : int or None, optional
            Divisor is (count - ddof). If None, uses instance default ``self.ddof``.

        Returns
        -------
        var : ndarray of shape (n_features,)
            Variance per feature (NaN if count <= ddof).
        """
        if self.n_features_ is None:
            return np.zeros(0, dtype=float)

        d = self.ddof if ddof is None else ddof
        if d < 0:
            raise ValueError(f"ddof must be non-negative, got {d}")

        denom = self._n - d
        return np.where(denom > 0, self._m2 / np.where(denom > 0, denom, 1.0), np.nan)

    def std(self, ddof: Optional[int] = None) -> np.ndarray:
        """Running standard deviation per feature."""
        return np.sqrt(self.var(ddof))

    def quantile(self, q: Union[float, Sequence[float]]) -> np.ndarray:
        """Compute exact quantiles across all observed non-NaN samples.

        Parameters
        ----------
        q : float or sequence of float
            Quantile or sequence of quantiles in [0.0, 1.0].

        Returns
        -------
        quantiles : ndarray of shape (n_features,) or (len(q), n_features)
            Per-feature quantiles.
        """
        if self.n_features_ is None or not self._sample_buffer:
            raise ValueError("No data has been observed yet.")

        q_arr = np.asarray(q, dtype=float)
        if np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
            raise ValueError(f"Quantile q must be in [0.0, 1.0], got {q!r}")

        all_samples = np.vstack(self._sample_buffer)
        return np.nanquantile(all_samples, q=q_arr, axis=0, method="linear")

    def histogram(
        self,
        bins: Union[int, Sequence[float], np.ndarray] = 10,
        range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute or retrieve streaming histogram over observed samples.

        Parameters
        ----------
        bins : int or array-like, optional
            Number of equal-width bins, or explicit bin edges. Default 10.
        range : (float, float) or None, optional
            Lower and upper range of bins. If None, derived from observed data.

        Returns
        -------
        counts : ndarray of shape (n_bins,)
            Bin frequencies.
        bin_edges : ndarray of shape (n_bins + 1,)
            Bin boundary edges.
        """
        if not self._sample_buffer:
            raise ValueError("No data has been observed yet.")

        all_samples = np.vstack(self._sample_buffer)
        flat_vals = all_samples[~np.isnan(all_samples)]
        if flat_vals.size == 0:
            raise ValueError("All observed values are NaN.")

        counts, bin_edges = np.histogram(flat_vals, bins=bins, range=range)
        self._bin_edges = bin_edges
        self._hist_counts = counts.copy()
        return counts, bin_edges

    def reset(self) -> "StreamingStats":
        """Reset internal accumulator state."""
        self.n_features_ = None
        self._n = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)
        self._sample_buffer.clear()
        self._bin_edges = None
        self._hist_counts = None
        return self


def update_stats(existing_stats: Optional[StreamingStats], X_chunk: Any) -> StreamingStats:
    """Convenience functional update for streaming statistics.

    Parameters
    ----------
    existing_stats : StreamingStats or None
        Accumulator instance to update, or None to create a new one.
    X_chunk : array-like
        Incoming data chunk.

    Returns
    -------
    stats : StreamingStats
        Updated statistics object.
    """
    if existing_stats is None:
        stats = StreamingStats()
    else:
        stats = existing_stats
    return stats.update(X_chunk)
