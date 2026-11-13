"""Tests for streaming preprocessing transformers: StandardScaler, Imputer, OneHotEncoder."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute.utils import NotFittedError
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler


# =====================================================================
# StandardScaler Tests
# =====================================================================
def test_standard_scaler_equivalence_to_batch():
    # Split data into 3 chunks
    chunk1 = np.array([[1.0, 10.0], [2.0, 20.0]])
    chunk2 = np.array([[3.0, 30.0], [4.0, 40.0]])
    chunk3 = np.array([[5.0, 50.0]])
    full_X = np.vstack([chunk1, chunk2, chunk3])

    scaler = StandardScaler()
    scaler.partial_fit(chunk1)
    scaler.partial_fit(chunk2)
    scaler.partial_fit(chunk3)

    assert scaler.n_samples_seen_ == 5
    np.testing.assert_allclose(scaler.mean_, np.mean(full_X, axis=0))
    np.testing.assert_allclose(scaler.var_, np.var(full_X, axis=0))
    np.testing.assert_allclose(scaler.scale_, np.std(full_X, axis=0))

    # Test transform
    transformed = scaler.transform(full_X)
    np.testing.assert_allclose(np.mean(transformed, axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.var(transformed, axis=0), [1.0, 1.0], atol=1e-12)


def test_standard_scaler_constant_zero_variance():
    # Feature 0 varies, feature 1 is constant 5.0
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = StandardScaler()
    scaler.fit(X)

    # Constant column must have scale_ == 1.0
    assert scaler.scale_[1] == 1.0
    assert scaler.var_[1] == 0.0

    # Transforming constant column yields zeros (5.0 - 5.0 = 0.0)
    transformed = scaler.transform(X)
    np.testing.assert_allclose(transformed[:, 1], [0.0, 0.0, 0.0])


def test_standard_scaler_preserves_nan():
    chunk = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0]])
    scaler = StandardScaler().partial_fit(chunk)

    transformed = scaler.transform(chunk)
    # Check that NaNs remain at the exact same indices
    assert np.isnan(transformed[0, 1])
    assert np.isnan(transformed[2, 0])
    assert not np.isnan(transformed[1, 0])
    assert not np.isnan(transformed[1, 1])


def test_standard_scaler_errors():
    scaler = StandardScaler()
    with pytest.raises(NotFittedError):
        scaler.transform([[1.0, 2.0]])

    scaler.fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="fitted with 2 features"):
        scaler.transform([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="fitted with 2 features"):
        scaler.partial_fit([[1.0, 2.0, 3.0]])

    # 1D input
    with pytest.raises(ValueError):
        scaler.fit([1.0, 2.0])

    # Infinity
    with pytest.raises(ValueError, match="infinity"):
        scaler.fit([[np.inf, 1.0]])


# =====================================================================
# Imputer Tests
# =====================================================================
def test_imputer_mean_strategy_across_chunks():
    chunk1 = np.array([[1.0, np.nan], [np.nan, 10.0]])
    chunk2 = np.array([[3.0, 20.0], [5.0, np.nan]])

    imputer = Imputer(strategy="mean")
    imputer.partial_fit(chunk1)
    imputer.partial_fit(chunk2)

    # Feature 0 observed values: 1.0, 3.0, 5.0 -> mean = 3.0
    # Feature 1 observed values: 10.0, 20.0 -> mean = 15.0
    np.testing.assert_allclose(imputer.statistics_, [3.0, 15.0])

    test_data = np.array([[np.nan, np.nan], [2.0, 4.0]])
    imputed = imputer.transform(test_data)
    np.testing.assert_allclose(imputed[0], [3.0, 15.0])
    np.testing.assert_allclose(imputed[1], [2.0, 4.0])


def test_imputer_all_nan_feature_fallback():
    # Feature 1 has all NaNs
    chunk = np.array([[1.0, np.nan], [2.0, np.nan]])
    imputer = Imputer(strategy="mean", fill_value=-999.0).fit(chunk)

    assert imputer.statistics_[1] == -999.0
    imputed = imputer.transform([[np.nan, np.nan]])
    np.testing.assert_allclose(imputed[0], [1.5, -999.0])


def test_imputer_constant_strategy():
    chunk = np.array([[1.0, np.nan], [2.0, 3.0]])
    imputer = Imputer(strategy="constant", fill_value=0.0).fit(chunk)
    imputed = imputer.transform([[np.nan, np.nan]])
    np.testing.assert_allclose(imputed[0], [0.0, 0.0])


def test_imputer_errors():
    with pytest.raises(ValueError, match="strategy"):
        Imputer(strategy="median")

    imputer = Imputer()
    with pytest.raises(NotFittedError):
        imputer.transform([[1.0, 2.0]])

    imputer.fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="fitted with 2 features"):
        imputer.transform([[1.0]])


# =====================================================================
# OneHotEncoder Tests
# =====================================================================
def test_onehot_encoder_incremental_categories():
    chunk1 = np.array([["apple"], ["banana"]])
    chunk2 = np.array([["banana"], ["cherry"]])

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.partial_fit(chunk1)
    assert len(encoder.categories_[0]) == 2
    np.testing.assert_array_equal(encoder.categories_[0], ["apple", "banana"])

    encoder.partial_fit(chunk2)
    assert len(encoder.categories_[0]) == 3
    # Alphabetically sorted unique categories
    np.testing.assert_array_equal(encoder.categories_[0], ["apple", "banana", "cherry"])

    # Transform
    encoded = encoder.transform([["banana"], ["cherry"], ["apple"]])
    expected = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(encoded, expected)


def test_onehot_encoder_unknown_categories_ignore():
    encoder = OneHotEncoder(handle_unknown="ignore").fit([["red"], ["blue"]])
    # "green" is unknown -> maps to all zeros
    encoded = encoder.transform([["red"], ["green"]])
    np.testing.assert_allclose(encoded[0], [0.0, 1.0])  # blue=0, red=1
    np.testing.assert_allclose(encoded[1], [0.0, 0.0])


def test_onehot_encoder_unknown_categories_error():
    encoder = OneHotEncoder(handle_unknown="error").fit([["red"], ["blue"]])
    with pytest.raises(ValueError, match="unknown category 'green'"):
        encoder.transform([["red"], ["green"]])


def test_onehot_encoder_multidimensional():
    X = np.array([
        ["cat", "male"],
        ["dog", "female"],
    ])
    encoder = OneHotEncoder().fit(X)
    assert len(encoder.categories_) == 2
    # Column 0: ['cat', 'dog'] (2 cols)
    # Column 1: ['female', 'male'] (2 cols)
    # Total encoded shape: (2, 4)
    encoded = encoder.transform(X)
    assert encoded.shape == (2, 4)
