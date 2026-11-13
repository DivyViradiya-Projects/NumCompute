"""Streaming Decision Tree Classifier.

Provides DecisionTreeClassifier with buffered retraining support, depth-limited
recursive splitting, deterministic tie-breaking, Gini / entropy criteria,
and pure-NumPy vectorised split evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.utils import NotFittedError, check_2d

__all__ = ["DecisionTreeClassifier"]


class _Node:
    """Internal decision tree node representation."""

    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "probabilities",
        "impurity",
        "n_samples",
        "is_leaf",
    )

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        value: Optional[int] = None,
        probabilities: Optional[np.ndarray] = None,
        impurity: float = 0.0,
        n_samples: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probabilities = probabilities
        self.impurity = impurity
        self.n_samples = n_samples
        self.is_leaf = is_leaf


def _compute_impurity(counts: np.ndarray, n_samples: int, criterion: str) -> float:
    """Compute impurity (Gini or entropy) from class counts.

    Parameters
    ----------
    counts : ndarray of int
        Count of samples per class.
    n_samples : int
        Total number of samples.
    criterion : {"gini", "entropy"}
        Impurity metric.

    Returns
    -------
    float
        Calculated node impurity.
    """
    if n_samples == 0:
        return 0.0
    probs = counts / n_samples
    if criterion == "gini":
        return float(1.0 - np.sum(probs**2))
    elif criterion == "entropy":
        # - sum(p * log2(p)), ignoring p == 0
        nonzero_p = probs[probs > 0]
        if nonzero_p.size == 0:
            return 0.0
        return float(-np.sum(nonzero_p * np.log2(nonzero_p)))
    else:
        raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")


def _evaluate_feature_splits(
    x_col: np.ndarray,
    y_encoded: np.ndarray,
    n_classes: int,
    criterion: str,
    parent_impurity: float,
) -> Tuple[float, Optional[float]]:
    """Find the best split threshold for a single numeric feature.

    Uses NumPy cumulative sums for O(N * C) vectorised split evaluation.

    Parameters
    ----------
    x_col : ndarray, shape (n_samples,)
        Feature values for the current node.
    y_encoded : ndarray, shape (n_samples,)
        Integer encoded labels in [0, n_classes - 1].
    n_classes : int
        Total number of distinct classes.
    criterion : {"gini", "entropy"}
        Split criterion.
    parent_impurity : float
        Impurity of parent node.

    Returns
    -------
    best_gain : float
        Maximum impurity gain found (0.0 if no valid split).
    best_threshold : float or None
        Optimal split threshold.
    """
    n_samples = len(x_col)
    if n_samples < 2:
        return 0.0, None

    # Sort values and labels along this feature
    order = np.argsort(x_col, kind="mergesort")
    x_sorted = x_col[order]
    y_sorted = y_encoded[order]

    # Find candidate split positions where adjacent feature values differ
    split_mask = x_sorted[:-1] != x_sorted[1:]
    split_indices = np.where(split_mask)[0]

    if split_indices.size == 0:
        return 0.0, None

    # One-hot encode sorted labels and compute cumulative counts: shape (n_samples, n_classes)
    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.int64)
    y_one_hot[np.arange(n_samples), y_sorted] = 1
    left_counts = np.cumsum(y_one_hot, axis=0)
    total_counts = left_counts[-1]

    # Only evaluate at valid split boundaries
    left_counts_splits = left_counts[split_indices]
    n_left_arr = split_indices + 1
    n_right_arr = n_samples - n_left_arr
    right_counts_splits = total_counts - left_counts_splits

    best_gain = 0.0
    best_threshold: Optional[float] = None

    # Evaluate candidate splits
    for idx, split_idx in enumerate(split_indices):
        n_left = int(n_left_arr[idx])
        n_right = int(n_right_arr[idx])
        if n_left == 0 or n_right == 0:
            continue

        left_cnt = left_counts_splits[idx]
        right_cnt = right_counts_splits[idx]

        imp_left = _compute_impurity(left_cnt, n_left, criterion)
        imp_right = _compute_impurity(right_cnt, n_right, criterion)

        weighted_imp = (n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right
        gain = parent_impurity - weighted_imp

        # Use midpoint as threshold
        threshold = float((x_sorted[split_idx] + x_sorted[split_idx + 1]) / 2.0)

        # Deterministic tie-breaking: prefer strictly higher gain, then lower threshold
        if gain > best_gain + 1e-12:
            best_gain = gain
            best_threshold = threshold
        elif abs(gain - best_gain) <= 1e-12 and best_threshold is not None:
            if threshold < best_threshold:
                best_threshold = threshold

    return best_gain, best_threshold


class DecisionTreeClassifier:
    """Decision Tree Classifier with buffered streaming retraining.

    Implements a recursive, depth-limited classification decision tree from scratch
    using pure NumPy. Supports incremental ingestion via ``partial_fit`` with
    strict buffered retraining semantics.

    Parameters
    ----------
    max_depth : int or None, optional
        Maximum tree depth. If None, nodes are expanded until pure or until
        fewer than ``min_samples_split`` samples remain. Default None.
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node. Must be >= 2.
        Default 2.
    max_features : int, float, {"sqrt", "log2"} or None, optional
        Number of features to consider when looking for the best split:
        - If int: consider ``max_features`` features directly (1 <= max_features <= n_features).
        - If float: consider ``max(1, int(max_features * n_features))`` features (0 < max_features <= 1.0).
        - If "sqrt": consider ``max(1, int(sqrt(n_features)))``.
        - If "log2": consider ``max(1, int(log2(n_features)))``.
        - If None: evaluate all features. Default None.
    criterion : {"gini", "entropy"}, optional
        Impurity measure to evaluate split quality. Default "gini".
    buffer_size : int, optional
        Number of buffered samples required to trigger a retrain during ``partial_fit``.
        Must be >= 1. Default 500.
    random_state : int or None, optional
        Seed for pseudo-random feature selection when ``max_features`` is set. Default None.

    Attributes
    ----------
    is_fitted_ : bool
        True if the tree has been trained at least once.
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels observed during the most recent fit.
    n_classes_ : int
        Number of unique classes.
    n_features_in_ : int
        Number of input features observed.
    n_retrains_ : int
        Total number of successful retraining operations executed.
    n_samples_seen_ : int
        Total number of streaming samples ingested via ``partial_fit``.
    tree_ : _Node or None
        Root node of the most recently fitted tree structure.
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

        self._validate_params()

        # Buffer state
        self._X_buffer: List[np.ndarray] = []
        self._y_buffer: List[np.ndarray] = []
        self._pending_count: int = 0

        # Model state
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_retrains_: int = 0
        self.n_samples_seen_: int = 0
        self.tree_: Optional[_Node] = None

    def _validate_params(self) -> None:
        """Validate constructor hyperparameter values."""
        if self.max_depth is not None:
            if (
                isinstance(self.max_depth, bool)
                or not isinstance(self.max_depth, (int, np.integer))
                or self.max_depth < 0
            ):
                raise ValueError(
                    f"max_depth must be None or a non-negative int, got {self.max_depth!r}"
                )

        if (
            isinstance(self.min_samples_split, bool)
            or not isinstance(self.min_samples_split, (int, np.integer))
            or self.min_samples_split < 2
        ):
            raise ValueError(
                f"min_samples_split must be an integer >= 2, got {self.min_samples_split!r}"
            )

        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {self.criterion!r}"
            )

        if (
            isinstance(self.buffer_size, bool)
            or not isinstance(self.buffer_size, (int, np.integer))
            or self.buffer_size < 1
        ):
            raise ValueError(
                f"buffer_size must be an integer >= 1, got {self.buffer_size!r}"
            )

        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features not in ("sqrt", "log2"):
                    raise ValueError(
                        f"max_features string must be 'sqrt' or 'log2', got {self.max_features!r}"
                    )
            elif isinstance(self.max_features, (int, np.integer)):
                if isinstance(self.max_features, bool) or self.max_features < 1:
                    raise ValueError(
                        f"max_features as int must be >= 1, got {self.max_features!r}"
                    )
            elif isinstance(self.max_features, (float, np.floating)):
                if not (0.0 < float(self.max_features) <= 1.0):
                    raise ValueError(
                        f"max_features as float must be in (0.0, 1.0], got {self.max_features!r}"
                    )
            else:
                raise TypeError(
                    f"max_features must be int, float, str or None, got {type(self.max_features).__name__}"
                )

    def _resolve_max_features(self, n_features: int) -> int:
        """Determine actual number of features to evaluate per split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, (int, np.integer)):
            if self.max_features > n_features:
                raise ValueError(
                    f"max_features ({self.max_features}) cannot exceed n_features ({n_features})"
                )
            return int(self.max_features)
        elif isinstance(self.max_features, (float, np.floating)):
            return max(1, int(float(self.max_features) * n_features))
        return n_features

    def _validate_data(
        self, X: Any, y: Optional[Any] = None, check_features: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate input feature matrix X and target vector y."""
        X_arr = check_2d(X, name="X", allow_empty=False)

        # Ensure float dtype for features
        if not np.issubdtype(X_arr.dtype, np.number):
            raise TypeError("X must contain numeric data")
        X_arr = X_arr.astype(float, copy=False)

        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or Inf values; impute or clean data first")

        if check_features and self.n_features_in_ is not None:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_arr.shape[1]} features, but {type(self).__name__} was fitted "
                    f"with {self.n_features_in_} features"
                )

        y_arr: Optional[np.ndarray] = None
        if y is not None:
            y_raw = np.asarray(y)
            if y_raw.ndim != 1:
                raise ValueError(f"y must be a 1-D array, got shape {y_raw.shape}")
            if len(y_raw) != len(X_arr):
                raise ValueError(
                    f"Number of samples in X ({len(X_arr)}) and y ({len(y_raw)}) must match"
                )
            if len(y_raw) == 0:
                raise ValueError("y must not be empty")
            if y_raw.dtype.kind == "f" and not np.all(np.isfinite(y_raw)):
                raise ValueError("y contains NaN or Inf values")
            y_arr = y_raw

        return X_arr, y_arr

    @property
    def n_buffered_samples_(self) -> int:
        """Current number of pending samples in the retraining buffer."""
        return self._pending_count

    def _build_tree(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        current_depth: int,
        k_features: int,
        rng: np.random.Generator,
    ) -> _Node:
        """Recursively construct decision tree nodes."""
        n_samples = len(X)
        n_features = X.shape[1]

        # Calculate class distribution at current node
        class_counts = np.bincount(y_encoded, minlength=self.n_classes_)
        probabilities = class_counts / n_samples
        # Deterministic leaf value: lowest index among majority classes
        majority_class_idx = int(np.argmax(class_counts))
        impurity = _compute_impurity(class_counts, n_samples, self.criterion)

        # Base leaf node representation
        leaf_node = _Node(
            value=majority_class_idx,
            probabilities=probabilities,
            impurity=impurity,
            n_samples=n_samples,
            is_leaf=True,
        )

        # Stopping conditions
        if (
            np.count_nonzero(class_counts) <= 1  # Pure node
            or (self.max_depth is not None and current_depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or impurity <= 1e-12
        ):
            return leaf_node

        # Feature selection
        if k_features < n_features:
            feature_indices = rng.choice(n_features, size=k_features, replace=False)
            feature_indices = np.sort(feature_indices)
        else:
            feature_indices = np.arange(n_features)

        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        # Search best split across candidate features
        for f_idx in feature_indices:
            gain, thresh = _evaluate_feature_splits(
                X[:, f_idx], y_encoded, self.n_classes_, self.criterion, impurity
            )
            # Deterministic tie-breaking: prefer higher gain, then lower feature index, then lower threshold
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = int(f_idx)
                best_threshold = thresh
            elif abs(gain - best_gain) <= 1e-12 and gain > 0 and best_feature is not None:
                if f_idx < best_feature:
                    best_feature = int(f_idx)
                    best_threshold = thresh
                elif f_idx == best_feature and thresh is not None and best_threshold is not None:
                    if thresh < best_threshold:
                        best_threshold = thresh

        if best_gain <= 1e-12 or best_feature is None or best_threshold is None:
            return leaf_node

        # Partition data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # If a child would be empty, do not split
        if not np.any(left_mask) or not np.any(right_mask):
            return leaf_node

        left_child = self._build_tree(
            X[left_mask], y_encoded[left_mask], current_depth + 1, k_features, rng
        )
        right_child = self._build_tree(
            X[right_mask], y_encoded[right_mask], current_depth + 1, k_features, rng
        )

        return _Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            value=majority_class_idx,
            probabilities=probabilities,
            impurity=impurity,
            n_samples=n_samples,
            is_leaf=False,
        )

    def _fit_tree(self, X: np.ndarray, y: np.ndarray) -> _Node:
        """Train a tree structure on validated arrays X and y."""
        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_features_in_ = X.shape[1]

        k_features = self._resolve_max_features(self.n_features_in_)
        rng = np.random.default_rng(self.random_state)

        return self._build_tree(
            X=X,
            y_encoded=y_encoded,
            current_depth=0,
            k_features=k_features,
            rng=rng,
        )

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        """Build a decision tree classifier from training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted classifier.
        """
        X_arr, y_arr = self._validate_data(X, y, check_features=False)
        assert y_arr is not None

        new_tree = self._fit_tree(X_arr, y_arr)

        self.tree_ = new_tree
        self.is_fitted_ = True
        self.n_retrains_ += 1
        return self

    def partial_fit(self, X_chunk: Any, y_chunk: Any) -> "DecisionTreeClassifier":
        """Incremental fit on a chunk of data with buffered retraining.

        Appends the incoming chunk to the pending buffer. When the buffer contains
        at least ``buffer_size`` samples, retrains on all currently pending samples
        (including any overshoot) and clears the buffer upon successful training.
        Predictions between retrains remain served by the prior fitted tree.

        Parameters
        ----------
        X_chunk : array-like of shape (n_chunk_samples, n_features)
            Incoming feature chunk.
        y_chunk : array-like of shape (n_chunk_samples,)
            Incoming target labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted classifier.
        """
        # If empty chunk, return self unchanged
        raw_X = np.asarray(X_chunk)
        if raw_X.size == 0 or len(raw_X) == 0:
            return self

        check_features = self.n_features_in_ is not None
        X_arr, y_arr = self._validate_data(X_chunk, y_chunk, check_features=check_features)
        assert y_arr is not None

        # Set n_features_in_ if first chunk
        if self.n_features_in_ is None:
            self.n_features_in_ = X_arr.shape[1]

        # Append chunk to pending buffer
        self._X_buffer.append(X_arr)
        self._y_buffer.append(y_arr)
        chunk_len = len(X_arr)
        self._pending_count += chunk_len
        self.n_samples_seen_ += chunk_len

        # Check if buffered count meets or exceeds buffer_size
        if self._pending_count >= self.buffer_size:
            X_train = np.vstack(self._X_buffer)
            y_train = np.concatenate(self._y_buffer)

            # Fit new tree; if it raises, buffer and old tree remain intact
            new_tree = self._fit_tree(X_train, y_train)

            # Retraining succeeded: commit new tree and clear buffer
            self.tree_ = new_tree
            self.is_fitted_ = True
            self.n_retrains_ += 1
            self._X_buffer.clear()
            self._y_buffer.clear()
            self._pending_count = 0

        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Force a retrain on any remaining pending samples in the buffer.

        If the buffer is empty, no retraining occurs.
        """
        if self._pending_count > 0:
            X_train = np.vstack(self._X_buffer)
            y_train = np.concatenate(self._y_buffer)
            new_tree = self._fit_tree(X_train, y_train)
            self.tree_ = new_tree
            self.is_fitted_ = True
            self.n_retrains_ += 1
            self._X_buffer.clear()
            self._y_buffer.clear()
            self._pending_count = 0
        return self

    def _predict_node(self, node: _Node, x: np.ndarray) -> _Node:
        """Traverse tree for a single sample x to find the terminal leaf."""
        curr = node
        while not curr.is_leaf:
            assert curr.feature is not None and curr.threshold is not None
            if x[curr.feature] <= curr.threshold:
                assert curr.left is not None
                curr = curr.left
            else:
                assert curr.right is not None
                curr = curr.right
        return curr

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for X using the most recently trained tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if not self.is_fitted_ or self.tree_ is None or self.classes_ is None:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")

        X_arr, _ = self._validate_data(X, check_features=True)
        n_samples = len(X_arr)
        proba = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i in range(n_samples):
            leaf = self._predict_node(self.tree_, X_arr[i])
            assert leaf.probabilities is not None
            proba[i] = leaf.probabilities

        return proba

    def predict(self, X: Any) -> np.ndarray:
        """Predict class for X using the most recently trained tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self.is_fitted_ or self.tree_ is None or self.classes_ is None:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")

        X_arr, _ = self._validate_data(X, check_features=True)
        n_samples = len(X_arr)
        y_indices = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            leaf = self._predict_node(self.tree_, X_arr[i])
            assert leaf.value is not None
            y_indices[i] = leaf.value

        return self.classes_[y_indices]

    def score(self, X: Any, y: Any) -> float:
        """Return mean accuracy on test data X and labels y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy in [0.0, 1.0].
        """
        y_pred = self.predict(X)
        y_arr = np.asarray(y)
        if len(y_arr) != len(y_pred):
            raise ValueError(
                f"y shape {y_arr.shape} does not match predictions shape {y_pred.shape}"
            )
        return float(np.mean(y_pred == y_arr))

    def get_depth(self) -> int:
        """Return the depth of the fitted tree.

        Depth is the maximum number of edges from root to any leaf. A tree with
        only a root leaf has depth 0.
        """
        if not self.is_fitted_ or self.tree_ is None:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")

        def _depth(node: _Node) -> int:
            if node.is_leaf:
                return 0
            left_d = _depth(node.left) if node.left else 0
            right_d = _depth(node.right) if node.right else 0
            return 1 + max(left_d, right_d)

        return _depth(self.tree_)

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes in the fitted tree."""
        if not self.is_fitted_ or self.tree_ is None:
            raise NotFittedError("This DecisionTreeClassifier instance is not fitted yet.")

        def _leaves(node: _Node) -> int:
            if node.is_leaf:
                return 1
            left_l = _leaves(node.left) if node.left else 0
            right_l = _leaves(node.right) if node.right else 0
            return left_l + right_l

        return _leaves(self.tree_)
