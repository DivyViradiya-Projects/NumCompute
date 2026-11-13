"""Incremental Preprocessing Transformers.

Provides streaming-compatible transformers:
- StandardScaler: Online mean and variance updates via Welford/Chan algorithm.
- Imputer: Online feature mean/mode missing value replacement.
- OneHotEncoder: Incremental category discovery with deterministic mapping.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.utils import NotFittedError, check_2d

__all__ = ["StandardScaler", "Imputer", "OneHotEncoder"]


class StandardScaler:
    """Standardize features by incrementally removing the mean and scaling to unit variance.

    Uses Chan et al.'s parallel / Welford update formulas to accumulate per-feature
    mean and variance across incoming streaming chunks without storing historical rows.

    Parameters
    ----------
    with_mean : bool, optional
        If True, center data before scaling. Default True.
    with_std : bool, optional
        If True, scale data to unit variance. Default True.

    Attributes
    ----------
    n_features_in_ : int or None
        Number of features seen during fitting.
    n_samples_seen_ : int
        Total number of non-empty samples processed across all chunks.
    mean_ : ndarray of shape (n_features,)
        Per-feature running mean.
    var_ : ndarray of shape (n_features,)
        Per-feature running population variance (ddof=0).
    scale_ : ndarray of shape (n_features,)
        Per-feature scaling divisor. Constant features get scale_ = 1.0.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        if not isinstance(with_mean, bool):
            raise TypeError(f"with_mean must be a bool, got {type(with_mean).__name__}")
        if not isinstance(with_std, bool):
            raise TypeError(f"with_std must be a bool, got {type(with_std).__name__}")
        self.with_mean = with_mean
        self.with_std = with_std

        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

        # Internal Welford / Chan accumulator state
        self._n = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)

    def _init_state(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        self._n = np.zeros(n_features, dtype=float)
        self._mean = np.zeros(n_features, dtype=float)
        self._m2 = np.zeros(n_features, dtype=float)

    def _validate_input(self, X: Any, allow_empty: bool = True) -> np.ndarray:
        arr = check_2d(X, name="X", allow_empty=allow_empty)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"X must be numeric, got dtype {arr.dtype}")
        arr_float = arr.astype(float, copy=False)
        if np.isinf(arr_float).any():
            raise ValueError("X contains infinity; clip or remove infinite values first")
        return arr_float

    def partial_fit(self, X: Any, y=None) -> "StandardScaler":
        """Incrementally update mean and variance estimates with chunk X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Incoming data chunk.
        y : ignored

        Returns
        -------
        self : StandardScaler
        """
        arr = self._validate_input(X, allow_empty=True)
        if arr.shape[0] == 0:
            return self

        if self.n_features_in_ is None:
            self._init_state(arr.shape[1])
        elif arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but {type(self).__name__} was fitted "
                f"with {self.n_features_in_} features"
            )

        # Non-NaN mask
        valid = ~np.isnan(arr)
        chunk_n = valid.sum(axis=0).astype(float)
        filled = np.where(valid, arr, 0.0)
        chunk_mean = filled.sum(axis=0) / np.maximum(chunk_n, 1.0)
        chunk_m2 = (((filled - chunk_mean) ** 2) * valid).sum(axis=0)

        # Merge with running state using Chan et al.'s formula
        total_n = self._n + chunk_n
        delta = chunk_mean - self._mean
        frac = chunk_n / np.maximum(total_n, 1.0)

        self._mean = self._mean + delta * frac
        self._m2 = self._m2 + chunk_m2 + (delta**2) * self._n * frac
        self._n = total_n

        self.n_samples_seen_ += arr.shape[0]

        # Update public attributes
        self.mean_ = np.where(self._n > 0, self._mean, 0.0)
        self.var_ = np.where(self._n > 0, self._m2 / np.maximum(self._n, 1.0), 0.0)

        # Constant columns (var == 0.0 or below machine precision) receive scale_ = 1.0
        eps = np.finfo(float).eps
        self.scale_ = np.where(self.var_ <= eps, 1.0, np.sqrt(np.maximum(self.var_, 0.0)))
        return self

    def fit(self, X: Any, y=None) -> "StandardScaler":
        """Fit StandardScaler on batch X (resets accumulator state).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : StandardScaler
        """
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self._n = np.zeros(0, dtype=float)
        self._mean = np.zeros(0, dtype=float)
        self._m2 = np.zeros(0, dtype=float)
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """Standardize X using current learned mean and scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Standardized data (preserves NaN positions).
        """
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise NotFittedError("This StandardScaler instance is not fitted yet.")

        arr = self._validate_input(X, allow_empty=True)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but StandardScaler was fitted "
                f"with {self.n_features_in_} features"
            )

        out = arr.copy()
        if self.with_mean:
            out -= self.mean_
        if self.with_std:
            out /= self.scale_
        return out

    def fit_transform(self, X: Any, y=None) -> np.ndarray:
        """Fit to batch X and return transformed array."""
        return self.fit(X, y).transform(X)


class Imputer:
    """Incremental missing value imputer.

    Tracks running column statistics across chunks and replaces NaN entries.

    Parameters
    ----------
    strategy : {"mean", "constant"}, optional
        Imputation strategy. Default "mean".
    fill_value : float, optional
        Replacement value when strategy="constant", or fallback when a feature
        has zero observed non-NaN samples. Default 0.0.

    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        Learned imputation value per feature.
    n_features_in_ : int or None
        Number of features seen.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in ("mean", "constant"):
            raise ValueError(f"strategy must be 'mean' or 'constant', got {strategy!r}")
        if not np.isfinite(fill_value):
            raise ValueError(f"fill_value must be finite, got {fill_value!r}")
        self.strategy = strategy
        self.fill_value = float(fill_value)

        self.n_features_in_: Optional[int] = None
        self.statistics_: Optional[np.ndarray] = None
        self._counts = np.zeros(0, dtype=float)
        self._sums = np.zeros(0, dtype=float)

    def _validate_input(self, X: Any, allow_empty: bool = True) -> np.ndarray:
        arr = check_2d(X, name="X", allow_empty=allow_empty)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"X must be numeric, got dtype {arr.dtype}")
        arr_float = arr.astype(float, copy=False)
        if np.isinf(arr_float).any():
            raise ValueError("X contains infinity; clip or remove infinite values first")
        return arr_float

    def partial_fit(self, X: Any, y=None) -> "Imputer":
        """Incrementally update missing value statistics with chunk X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Incoming data chunk.
        y : ignored

        Returns
        -------
        self : Imputer
        """
        arr = self._validate_input(X, allow_empty=True)
        if arr.shape[0] == 0:
            return self

        if self.n_features_in_ is None:
            self.n_features_in_ = arr.shape[1]
            self._counts = np.zeros(arr.shape[1], dtype=float)
            self._sums = np.zeros(arr.shape[1], dtype=float)
        elif arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but Imputer was fitted "
                f"with {self.n_features_in_} features"
            )

        if self.strategy == "mean":
            valid = ~np.isnan(arr)
            self._counts += valid.sum(axis=0)
            self._sums += np.where(valid, arr, 0.0).sum(axis=0)
            stats = np.where(
                self._counts > 0,
                self._sums / np.maximum(self._counts, 1.0),
                self.fill_value,
            )
            self.statistics_ = stats
        elif self.strategy == "constant":
            self.statistics_ = np.full(self.n_features_in_, self.fill_value, dtype=float)

        return self

    def fit(self, X: Any, y=None) -> "Imputer":
        """Fit Imputer on batch X (resets internal statistics).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : Imputer
        """
        self.n_features_in_ = None
        self.statistics_ = None
        self._counts = np.zeros(0, dtype=float)
        self._sums = np.zeros(0, dtype=float)
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """Impute missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Copy of X with NaN replaced by learned statistics.
        """
        if self.statistics_ is None or self.n_features_in_ is None:
            raise NotFittedError("This Imputer instance is not fitted yet.")

        arr = self._validate_input(X, allow_empty=True)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but Imputer was fitted "
                f"with {self.n_features_in_} features"
            )

        out = arr.copy()
        nan_mask = np.isnan(out)
        if nan_mask.any():
            out = np.where(nan_mask, self.statistics_, out)
        return out

    def fit_transform(self, X: Any, y=None) -> np.ndarray:
        """Fit and transform batch X."""
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    """Incremental One-Hot Encoder with category discovery.

    Discovers categories incrementally across chunks and provides a deterministic,
    stable column mapping.

    Parameters
    ----------
    handle_unknown : {"ignore", "error"}, optional
        Whether to raise an error or ignore (produce all-zero row) when unknown
        categories are encountered during transform. Default "ignore".

    Attributes
    ----------
    categories_ : list of ndarray
        Sorted unique categories discovered per feature column.
    n_features_in_ : int or None
        Number of input features.
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                f"handle_unknown must be 'ignore' or 'error', got {handle_unknown!r}"
            )
        self.handle_unknown = handle_unknown

        self.categories_: Optional[List[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None

    def _prepare_data(self, X: Any) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {arr.shape}")
        if arr.dtype.kind == "O":
            raise TypeError("Object arrays not supported; use strings, numbers or booleans")
        return arr

    def partial_fit(self, X: Any, y=None) -> "OneHotEncoder":
        """Incrementally discover new categories from chunk X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Incoming chunk.
        y : ignored

        Returns
        -------
        self : OneHotEncoder
        """
        arr = self._prepare_data(X)
        if arr.shape[0] == 0:
            return self

        n_cols = arr.shape[1]
        if self.n_features_in_ is None:
            self.n_features_in_ = n_cols
            self.categories_ = []
            for col_idx in range(n_cols):
                unique_vals = np.unique(arr[:, col_idx])
                self.categories_.append(unique_vals)
        else:
            if n_cols != self.n_features_in_:
                raise ValueError(
                    f"X has {n_cols} features, but OneHotEncoder was fitted with "
                    f"{self.n_features_in_} features"
                )
            assert self.categories_ is not None
            for col_idx in range(n_cols):
                existing = self.categories_[col_idx]
                incoming = np.unique(arr[:, col_idx])
                merged = np.unique(np.concatenate([existing, incoming]))
                self.categories_[col_idx] = merged

        return self

    def fit(self, X: Any, y=None) -> "OneHotEncoder":
        """Fit OneHotEncoder on batch X (resets categories).

        Parameters
        ----------
        X : array-like
            Training data.
        y : ignored

        Returns
        -------
        self : OneHotEncoder
        """
        self.n_features_in_ = None
        self.categories_ = None
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """One-hot encode X using learned categories.

        Parameters
        ----------
        X : array-like
            Data to encode.

        Returns
        -------
        X_encoded : ndarray of shape (n_samples, total_categories)
            Encoded float matrix.
        """
        if self.categories_ is None or self.n_features_in_ is None:
            raise NotFittedError("This OneHotEncoder instance is not fitted yet.")

        arr = self._prepare_data(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but OneHotEncoder was fitted with "
                f"{self.n_features_in_} features"
            )

        n_samples = arr.shape[0]
        if n_samples == 0:
            total_cols = sum(len(c) for c in self.categories_)
            return np.zeros((0, total_cols), dtype=float)

        encoded_blocks: List[np.ndarray] = []
        for col_idx in range(self.n_features_in_):
            cats = self.categories_[col_idx]
            n_cats = len(cats)
            block = np.zeros((n_samples, n_cats), dtype=float)
            col_vals = arr[:, col_idx]

            for cat_idx, cat_val in enumerate(cats):
                matches = (col_vals == cat_val)
                block[matches, cat_idx] = 1.0

            if self.handle_unknown == "error":
                known_mask = block.sum(axis=1) > 0
                if not np.all(known_mask):
                    unknown_val = str(col_vals[~known_mask][0])
                    raise ValueError(
                        f"Found unknown category {unknown_val!r} in column {col_idx}"
                    )

            encoded_blocks.append(block)

        return np.hstack(encoded_blocks)

    def fit_transform(self, X: Any, y=None) -> np.ndarray:
        """Fit and transform batch X."""
        return self.fit(X, y).transform(X)
