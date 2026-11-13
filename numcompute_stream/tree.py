"""Streaming Decision Tree Classifier.

Provides DecisionTreeClassifier with buffered retraining support, depth-limited
recursive splitting, deterministic tie-breaking, and Gini / entropy criteria.
"""
from __future__ import annotations

from typing import Optional, Union
import numpy as np

from numcompute.utils import NotFittedError

__all__ = ["DecisionTreeClassifier"]


class DecisionTreeClassifier:
    """Decision Tree Classifier with buffered streaming retraining.

    Parameters
    ----------
    max_depth : int or None, optional
        Maximum depth of the tree. If None, nodes are expanded until pure
        or fewer than min_samples_split samples remain. Default None.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node. Default 2.
    max_features : int, float, str, or None, optional
        Number of features to consider when looking for the best split:
        - If int, consider ``max_features`` features at each split.
        - If float, consider ``int(max_features * n_features)`` features.
        - If "sqrt", consider ``max(1, int(sqrt(n_features)))``.
        - If "log2", consider ``max(1, int(log2(n_features)))``.
        - If None, consider all features. Default None.
    criterion : {"gini", "entropy"}, optional
        Function to measure split quality. Default "gini".
    buffer_size : int, optional
        Number of pending samples required to trigger a retrain during
        ``partial_fit``. Default 500.
    random_state : int or None, optional
        Seed for random feature selection when ``max_features`` is used.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[Union[int, float, str]] = None,
        criterion: str = "gini",
        buffer_size: int = 500,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.random_state = random_state

        # Buffer state
        self._X_buffer: list[np.ndarray] = []
        self._y_buffer: list[np.ndarray] = []
        self._pending_count: int = 0

        # Model state
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_retrains_: int = 0
        self.n_samples_seen_: int = 0
        self.tree_: Optional[object] = None

    @property
    def n_buffered_samples_(self) -> int:
        """Current number of pending samples in the retraining buffer."""
        return self._pending_count

    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Build a decision tree classifier from training set (X, y)."""
        raise NotImplementedError("Will be implemented in Phase 2")

    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """Incremental fit on a chunk of data with buffered retraining."""
        raise NotImplementedError("Will be implemented in Phase 2")

    def predict(self, X) -> np.ndarray:
        """Predict class for X using the most recently trained tree."""
        if not self.is_fitted_:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 2")

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X using the most recently trained tree."""
        if not self.is_fitted_:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 2")

    def score(self, X, y) -> float:
        """Return mean accuracy on test data and labels."""
        raise NotImplementedError("Will be implemented in Phase 2")

    def get_depth(self) -> int:
        """Return the depth of the fitted tree."""
        if not self.is_fitted_:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 2")

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the fitted tree."""
        if not self.is_fitted_:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 2")
