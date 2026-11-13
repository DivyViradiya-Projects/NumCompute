"""Incremental Preprocessing Transformers.

Provides streaming versions of StandardScaler, Imputer, and OneHotEncoder
supporting ``partial_fit`` and ``transform``.
"""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from numcompute.utils import NotFittedError

__all__ = ["StandardScaler", "Imputer", "OneHotEncoder"]


class StandardScaler:
    """Incremental Standard Scaler using Welford / Chan running mean and variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.n_features_in_: Optional[int] = None
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def fit(self, X, y=None) -> "StandardScaler":
        """Compute mean and std on batch X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Incrementally update mean and std with chunk X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def transform(self, X) -> np.ndarray:
        """Standardize X using current running statistics."""
        if self.mean_ is None:
            raise NotFittedError("StandardScaler is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 3")

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform batch X."""
        return self.fit(X, y).transform(X)


class Imputer:
    """Incremental Imputer supporting mean or most_frequent strategy."""

    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None) -> "Imputer":
        """Compute imputation statistics on batch X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def partial_fit(self, X, y=None) -> "Imputer":
        """Incrementally update imputation statistics with chunk X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def transform(self, X) -> np.ndarray:
        """Impute missing values in X."""
        if self.statistics_ is None:
            raise NotFittedError("Imputer is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 3")

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform batch X."""
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    """Incremental OneHotEncoder with category discovery and stable mappings."""

    def __init__(self, handle_unknown: str = "ignore") -> None:
        self.handle_unknown = handle_unknown
        self.categories_: Optional[list[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None) -> "OneHotEncoder":
        """Learn categories from batch X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Incrementally discover categories from chunk X."""
        raise NotImplementedError("Will be implemented in Phase 3")

    def transform(self, X) -> np.ndarray:
        """One-hot encode X using learned categories."""
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder is not fitted yet.")
        raise NotImplementedError("Will be implemented in Phase 3")

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform batch X."""
        return self.fit(X, y).transform(X)
