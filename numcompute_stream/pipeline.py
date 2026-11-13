"""Streaming Pipeline.

Chains transformers with a final estimator, orchestrating incremental updates
and managing a raw-feature buffer with transformer snapshots to ensure 100%
consistent feature coordinates during both inference and retraining.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from numcompute.utils import NotFittedError, check_2d

__all__ = ["Pipeline"]


class Pipeline:
    """Streaming Pipeline with raw-pending buffer and committed transformer snapshots.

    Solves feature coordinate drift by maintaining a committed transformer snapshot
    paired with the fitted estimator. Incoming chunks update candidate transformers
    and accumulate raw samples. When the buffer reaches ``buffer_size``, all buffered
    samples are transformed under the up-to-date candidate transformers, the estimator
    is retrained, and the new transformer state is atomically committed. Inference
    between retrains uses the committed snapshot, guaranteeing that feature
    transformations always match the coordinate system in which the tree was trained.

    Parameters
    ----------
    steps : sequence of (name, object)
        Sequential pipeline stages. All steps except the last must be transformers
        implementing ``transform`` (and optionally ``partial_fit`` or ``fit``).
        The final step must be an estimator implementing ``fit`` and ``predict``.
    buffer_size : int or None, optional
        Retraining sample threshold. If None, inherits ``buffer_size`` from the
        final estimator if available, or defaults to 500.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the pipeline and final estimator have been trained at least once.
    n_buffered_samples_ : int
        Current number of pending samples in the raw feature buffer.
    n_samples_seen_ : int
        Total number of samples ingested via ``partial_fit``.
    n_retrains_ : int
        Total number of retrainings executed on the final estimator.
    """

    def __init__(
        self,
        steps: Sequence[Tuple[str, Any]],
        buffer_size: Optional[int] = None,
    ) -> None:
        self.steps = list(steps)
        self._validate_steps()

        # Resolve buffer size
        final_est = self.steps[-1][1]
        if buffer_size is not None:
            if isinstance(buffer_size, bool) or not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
                raise ValueError(f"buffer_size must be an integer >= 1, got {buffer_size!r}")
            self.buffer_size = int(buffer_size)
        elif hasattr(final_est, "buffer_size"):
            est_bs = getattr(final_est, "buffer_size")
            if isinstance(est_bs, bool) or not isinstance(est_bs, (int, np.integer)) or est_bs < 1:
                raise ValueError(f"Inherited buffer_size must be an integer >= 1, got {est_bs!r}")
            self.buffer_size = int(est_bs)
        else:
            self.buffer_size = 500

        # Raw sample buffer to preserve feature representation consistency
        self._raw_X_buffer: List[np.ndarray] = []
        self._raw_y_buffer: List[np.ndarray] = []
        self._pending_count: int = 0

        # Committed snapshot of intermediate transformers matching the fitted estimator
        self._committed_transformers: Optional[List[Tuple[str, Any]]] = None

        # State tracking
        self.is_fitted_: bool = False
        self.n_samples_seen_: int = 0
        self.n_retrains_: int = 0

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("steps must not be empty")

        names = []
        for idx, item in enumerate(self.steps):
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise TypeError(
                    f"Each step must be a (name, object) tuple, got {item!r} at index {idx}"
                )
            name, step = item
            if len(name.strip()) == 0:
                raise ValueError(f"Step name at index {idx} must be non-empty")
            names.append(name)

            is_last = idx == len(self.steps) - 1
            if not is_last:
                if not hasattr(step, "transform"):
                    raise TypeError(
                        f"Step {name!r} at index {idx} must implement 'transform', "
                        f"got {type(step).__name__}"
                    )
            else:
                if not hasattr(step, "fit") or not hasattr(step, "predict"):
                    raise TypeError(
                        f"Final step {name!r} must implement 'fit' and 'predict', "
                        f"got {type(step).__name__}"
                    )

        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got duplicate names in {names}")

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline stages by name as a dictionary."""
        return dict(self.steps)

    @property
    def n_buffered_samples_(self) -> int:
        """Current number of pending samples in the raw buffer."""
        return self._pending_count

    def _transform_steps(self, transformer_steps: List[Tuple[str, Any]], X: Any) -> np.ndarray:
        """Transform X sequentially through a specific set of transformers."""
        curr_X = X
        for name, transformer in transformer_steps:
            curr_X = transformer.transform(curr_X)
        return curr_X

    def fit(self, X: Any, y: Any) -> "Pipeline":
        """Fit all transformers and the final estimator on batch (X, y).

        Parameters
        ----------
        X : array-like
            Training feature data.
        y : array-like
            Target labels.

        Returns
        -------
        self : Pipeline
        """
        curr_X = X
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, "fit_transform"):
                curr_X = transformer.fit_transform(curr_X)
            elif hasattr(transformer, "fit"):
                transformer.fit(curr_X)
                curr_X = transformer.transform(curr_X)
            else:
                curr_X = transformer.transform(curr_X)

        final_est = self.steps[-1][1]
        final_est.fit(curr_X, y)

        # Snapshot intermediate transformers
        self._committed_transformers = copy.deepcopy(self.steps[:-1])
        self._raw_X_buffer.clear()
        self._raw_y_buffer.clear()
        self._pending_count = 0
        self.is_fitted_ = True
        self.n_retrains_ += 1
        return self

    def partial_fit(self, X_chunk: Any, y_chunk: Any) -> "Pipeline":
        """Process an incoming streaming chunk.

        Updates candidate transformer statistics incrementally and stores the raw
        incoming samples in the pipeline's raw buffer. When the buffer reaches
        ``buffer_size``, the entire raw buffer is transformed using the unified
        candidate transformer state, used to retrain the final estimator via ``fit``,
        and atomically committed.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Incoming feature chunk.
        y_chunk : array-like of shape (n_samples,)
            Incoming target labels.

        Returns
        -------
        self : Pipeline
        """
        raw_X = np.asarray(X_chunk)
        raw_y = np.asarray(y_chunk)

        if raw_X.size == 0 or len(raw_X) == 0:
            return self

        if len(raw_X) != len(raw_y):
            raise ValueError(
                f"X_chunk length ({len(raw_X)}) and y_chunk length ({len(raw_y)}) must match"
            )

        # Update candidate intermediate transformers chunk-by-chunk
        curr_X = raw_X
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(curr_X)
            elif hasattr(transformer, "fit") and not getattr(transformer, "is_fitted_", False):
                transformer.fit(curr_X)

            curr_X = transformer.transform(curr_X)

        # Retain raw chunks in pending buffer
        self._raw_X_buffer.append(raw_X.copy())
        self._raw_y_buffer.append(raw_y.copy())
        chunk_len = len(raw_X)
        self._pending_count += chunk_len
        self.n_samples_seen_ += chunk_len

        # Check if buffered count triggers retraining
        if self._pending_count >= self.buffer_size:
            self._retrain_estimator()

        return self

    def _retrain_estimator(self) -> None:
        """Transform complete raw pending window under latest candidate transformers and fit estimator."""
        if self._pending_count == 0:
            return

        X_raw_all = np.vstack(self._raw_X_buffer)
        y_raw_all = np.concatenate(self._raw_y_buffer)

        # Transform entire training window in unified current candidate space
        X_trans = self._transform_steps(self.steps[:-1], X_raw_all)

        final_est = self.steps[-1][1]
        # Train estimator on the fully transformed raw window
        final_est.fit(X_trans, y_raw_all)

        # Succeeded: atomically commit transformer snapshot and clear raw buffer
        self._committed_transformers = copy.deepcopy(self.steps[:-1])
        self.is_fitted_ = True
        self.n_retrains_ += 1
        self._raw_X_buffer.clear()
        self._raw_y_buffer.clear()
        self._pending_count = 0

    def flush(self) -> "Pipeline":
        """Force a retrain on any remaining pending samples in the buffer."""
        if self._pending_count > 0:
            self._retrain_estimator()
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Transform X using the committed transformer snapshot and predict with the final estimator.

        Parameters
        ----------
        X : array-like
            Query samples.

        Returns
        -------
        y_pred : ndarray
            Predicted class labels.
        """
        if not self.is_fitted_ or self._committed_transformers is None:
            raise NotFittedError("This Pipeline instance is not fitted yet.")

        X_trans = self._transform_steps(self._committed_transformers, X)
        final_est = self.steps[-1][1]
        return final_est.predict(X_trans)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Transform X using the committed transformer snapshot and return class probabilities.

        Parameters
        ----------
        X : array-like
            Query samples.

        Returns
        -------
        proba : ndarray
            Predicted probabilities.
        """
        if not self.is_fitted_ or self._committed_transformers is None:
            raise NotFittedError("This Pipeline instance is not fitted yet.")

        X_trans = self._transform_steps(self._committed_transformers, X)
        final_est = self.steps[-1][1]
        if hasattr(final_est, "predict_proba"):
            return final_est.predict_proba(X_trans)
        raise AttributeError(f"{type(final_est).__name__} does not implement predict_proba")

    def score(self, X: Any, y: Any) -> float:
        """Return mean prediction accuracy on test data (X, y)."""
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
        y_pred = self.predict(X)
        if len(y_arr) != len(y_pred):
            raise ValueError(f"Lengths do not match: len(y)={len(y_arr)} vs len(y_pred)={len(y_pred)}")
        return float(np.mean(y_pred == y_arr))
