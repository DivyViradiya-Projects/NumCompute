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


def _safe_midpoint(a: float, b: float) -> float:
    """Compute overflow-safe split threshold between two adjacent distinct floats a < b."""
    # Try halved sum: never overflows for finite floats
    mid = float(a / 2.0 + b / 2.0)
    if a <= mid < b:
        return mid
    # Fallback to offset
    diff_mid = float(a + (b - a) / 2.0)
    if a <= diff_mid < b:
        return diff_mid
    # If floating point precision cannot represent a value strictly between a and b,
    # choosing a ensures that 'x <= a' puts a in the left child and b in the right child.
    return float(a)


def _compute_impurity(counts: np.ndarray, n_samples: int, criterion: str) -> float:
    """Compute impurity (Gini or entropy) from class counts."""
    if n_samples == 0:
        return 0.0
    probs = counts / n_samples
    if criterion == "gini":
        return float(1.0 - np.sum(probs**2))
    elif criterion == "entropy":
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

    Uses vectorised NumPy cumulative counts and batch impurity evaluation.
    """
    n_samples = len(x_col)
    if n_samples < 2:
        return 0.0, None

    # Stable sort along feature values
    order = np.argsort(x_col, kind="mergesort")
    x_sorted = x_col[order]
    y_sorted = y_encoded[order]

    # Find candidate split positions where adjacent feature values differ
    split_mask = x_sorted[:-1] != x_sorted[1:]
    split_indices = np.where(split_mask)[0]

    if split_indices.size == 0:
        return 0.0, None

    # Cumulative class counts across sorted samples
    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.int64)
    y_one_hot[np.arange(n_samples), y_sorted] = 1
    left_counts = np.cumsum(y_one_hot, axis=0)
    total_counts = left_counts[-1]

    left_counts_splits = left_counts[split_indices].astype(float)
    n_left_arr = (split_indices + 1).astype(float)
    n_right_arr = (n_samples - n_left_arr).astype(float)
    right_counts_splits = total_counts.astype(float) - left_counts_splits

    # Compute child class probabilities across all candidate splits simultaneously
    p_left = left_counts_splits / n_left_arr[:, None]
    p_right = right_counts_splits / n_right_arr[:, None]

    if criterion == "gini":
        imp_left = 1.0 - np.sum(p_left**2, axis=1)
        imp_right = 1.0 - np.sum(p_right**2, axis=1)
    elif criterion == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p_l = np.where(p_left > 0, np.log2(p_left), 0.0)
            log_p_r = np.where(p_right > 0, np.log2(p_right), 0.0)
        imp_left = -np.sum(p_left * log_p_l, axis=1)
        imp_right = -np.sum(p_right * log_p_r, axis=1)
    else:
        raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")

    weighted_imp = (n_left_arr / n_samples) * imp_left + (n_right_arr / n_samples) * imp_right
    gains = parent_impurity - weighted_imp

    # Candidate thresholds using safe midpoint
    thresholds = np.array([
        _safe_midpoint(x_sorted[idx], x_sorted[idx + 1]) for idx in split_indices
    ])

    valid_gains_mask = gains > 1e-12
    if not np.any(valid_gains_mask):
        return 0.0, None

    max_gain = float(np.max(gains))
    if max_gain <= 1e-12:
        return 0.0, None

    # Deterministic tie-breaking: choose lowest threshold among optimal gains
    best_gain_indices = np.where(np.abs(gains - max_gain) <= 1e-12)[0]
    best_split_idx = best_gain_indices[np.argmin(thresholds[best_gain_indices])]

    return float(gains[best_split_idx]), float(thresholds[best_split_idx])


class DecisionTreeClassifier:
    """Decision Tree Classifier with buffered streaming retraining.

    Implements a recursive, depth-limited classification decision tree from scratch
    using pure NumPy. Supports incremental ingestion via ``partial_fit`` with
    strict buffered retraining semantics and atomic state updates.
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

        if self.random_state is not None:
            if isinstance(self.random_state, bool) or not isinstance(
                self.random_state, (int, np.integer)
            ):
                raise ValueError(
                    f"random_state must be None or an integer, got {self.random_state!r}"
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
        n_classes: int,
        rng: np.random.Generator,
    ) -> _Node:
        """Recursively construct decision tree nodes."""
        n_samples = len(X)
        n_features = X.shape[1]

        class_counts = np.bincount(y_encoded, minlength=n_classes)
        probabilities = class_counts / n_samples
        majority_class_idx = int(np.argmax(class_counts))
        impurity = _compute_impurity(class_counts, n_samples, self.criterion)

        leaf_node = _Node(
            value=majority_class_idx,
            probabilities=probabilities,
            impurity=impurity,
            n_samples=n_samples,
            is_leaf=True,
        )

        if (
            np.count_nonzero(class_counts) <= 1
            or (self.max_depth is not None and current_depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or impurity <= 1e-12
        ):
            return leaf_node

        if k_features < n_features:
            feature_indices = rng.choice(n_features, size=k_features, replace=False)
            feature_indices = np.sort(feature_indices)
        else:
            feature_indices = np.arange(n_features)

        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for f_idx in feature_indices:
            gain, thresh = _evaluate_feature_splits(
                X[:, f_idx], y_encoded, n_classes, self.criterion, impurity
            )
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

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return leaf_node

        left_child = self._build_tree(
            X[left_mask], y_encoded[left_mask], current_depth + 1, k_features, n_classes, rng
        )
        right_child = self._build_tree(
            X[right_mask], y_encoded[right_mask], current_depth + 1, k_features, n_classes, rng
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

    def _fit_candidate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[_Node, np.ndarray, int, int]:
        """Fit a candidate tree in local variables without mutating self (atomic)."""
        classes, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        n_features = X.shape[1]

        k_features = self._resolve_max_features(n_features)
        rng = np.random.default_rng(self.random_state)

        root = self._build_tree(
            X=X,
            y_encoded=y_encoded,
            current_depth=0,
            k_features=k_features,
            n_classes=n_classes,
            rng=rng,
        )
        return root, classes, n_classes, n_features

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        """Build a decision tree classifier from batch (X, y).

        Resets any pending streaming buffer from prior ``partial_fit`` sessions.

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

        new_tree, new_classes, new_n_classes, new_n_features = self._fit_candidate(
            X_arr, y_arr
        )

        # Commit state together and reset streaming buffer
        self.tree_ = new_tree
        self.classes_ = new_classes
        self.n_classes_ = new_n_classes
        self.n_features_in_ = new_n_features
        self.is_fitted_ = True
        self.n_retrains_ += 1

        self._X_buffer.clear()
        self._y_buffer.clear()
        self._pending_count = 0
        return self

    def partial_fit(self, X_chunk: Any, y_chunk: Any) -> "DecisionTreeClassifier":
        """Incremental fit on a chunk of data with buffered retraining.

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
        raw_X = np.asarray(X_chunk)
        if raw_X.size == 0 or len(raw_X) == 0:
            return self

        check_features = self.n_features_in_ is not None
        X_arr, y_arr = self._validate_data(X_chunk, y_chunk, check_features=check_features)
        assert y_arr is not None

        # Tentative feature count
        orig_features_in = self.n_features_in_
        if self.n_features_in_ is None:
            self.n_features_in_ = X_arr.shape[1]

        self._X_buffer.append(X_arr)
        self._y_buffer.append(y_arr)
        chunk_len = len(X_arr)
        self._pending_count += chunk_len
        self.n_samples_seen_ += chunk_len

        # Check if buffered count meets or exceeds buffer_size
        if self._pending_count >= self.buffer_size:
            X_train = np.vstack(self._X_buffer)
            y_train = np.concatenate(self._y_buffer)

            try:
                new_tree, new_classes, new_n_classes, new_n_features = self._fit_candidate(
                    X_train, y_train
                )
            except Exception:
                # If training failed on an unfitted model, revert n_features_in_
                if not self.is_fitted_:
                    self.n_features_in_ = orig_features_in
                raise

            # Atomic commit of new model state and buffer clear
            self.tree_ = new_tree
            self.classes_ = new_classes
            self.n_classes_ = new_n_classes
            self.n_features_in_ = new_n_features
            self.is_fitted_ = True
            self.n_retrains_ += 1
            self._X_buffer.clear()
            self._y_buffer.clear()
            self._pending_count = 0

        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Force a retrain on any remaining pending samples in the buffer."""
        if self._pending_count > 0:
            X_train = np.vstack(self._X_buffer)
            y_train = np.concatenate(self._y_buffer)
            orig_features_in = self.n_features_in_
            try:
                new_tree, new_classes, new_n_classes, new_n_features = self._fit_candidate(
                    X_train, y_train
                )
            except Exception:
                if not self.is_fitted_:
                    self.n_features_in_ = orig_features_in
                raise
            self.tree_ = new_tree
            self.classes_ = new_classes
            self.n_classes_ = new_n_classes
            self.n_features_in_ = new_n_features
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
        """Predict class probabilities for X using the most recently trained tree."""
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
        """Predict class for X using the most recently trained tree."""
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
        """Return mean accuracy on test data X and labels y."""
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be a 1-D array, got shape {y_arr.shape}")
        y_pred = self.predict(X)
        if len(y_arr) != len(y_pred):
            raise ValueError(
                f"y shape {y_arr.shape} does not match predictions shape {y_pred.shape}"
            )
        return float(np.mean(y_pred == y_arr))

    def get_depth(self) -> int:
        """Return the depth of the fitted tree."""
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
