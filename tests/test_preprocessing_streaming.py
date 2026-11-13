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

    transformed = scaler.transform(full_X)
    np.testing.assert_allclose(np.mean(transformed, axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.var(transformed, axis=0), [1.0, 1.0], atol=1e-12)


def test_standard_scaler_constant_zero_variance():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = StandardScaler()
    scaler.fit(X)

    assert scaler.scale_[1] == 1.0
    assert scaler.var_[1] == 0.0

    transformed = scaler.transform(X)
    np.testing.assert_allclose(transformed[:, 1], [0.0, 0.0, 0.0])


def test_standard_scaler_preserves_nan():
    chunk = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0]])
    scaler = StandardScaler().partial_fit(chunk)

    transformed = scaler.transform(chunk)
    assert np.isnan(transformed[0, 1])
    assert np.isnan(transformed[2, 0])
    assert not np.isnan(transformed[1, 0])
    assert not np.isnan(transformed[1, 1])


def test_standard_scaler_fit_resets_state():
    scaler = StandardScaler().partial_fit([[100.0, 200.0]])
    assert scaler.n_samples_seen_ == 1

    # fit on fresh data resets everything
    scaler.fit([[1.0, 2.0], [3.0, 4.0]])
    assert scaler.n_samples_seen_ == 2
    np.testing.assert_allclose(scaler.mean_, [2.0, 3.0])


def test_standard_scaler_errors():
    with pytest.raises(TypeError, match="with_mean must be a bool"):
        StandardScaler(with_mean="true")

    with pytest.raises(TypeError, match="with_std must be a bool"):
        StandardScaler(with_std=1)

    scaler = StandardScaler()
    with pytest.raises(NotFittedError):
        scaler.transform([[1.0, 2.0]])

    scaler.fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="fitted with 2 features"):
        scaler.transform([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="fitted with 2 features"):
        scaler.partial_fit([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        scaler.fit([1.0, 2.0])

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

    np.testing.assert_allclose(imputer.statistics_, [3.0, 15.0])

    test_data = np.array([[np.nan, np.nan], [2.0, 4.0]])
    imputed = imputer.transform(test_data)
    np.testing.assert_allclose(imputed[0], [3.0, 15.0])
    np.testing.assert_allclose(imputed[1], [2.0, 4.0])


def test_imputer_all_nan_feature_fallback():
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


def test_imputer_fit_resets_state():
    imputer = Imputer(strategy="mean").partial_fit([[100.0, 200.0]])
    imputer.fit([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(imputer.statistics_, [2.0, 3.0])


def test_imputer_errors():
    with pytest.raises(ValueError, match="strategy"):
        Imputer(strategy="median")

    with pytest.raises(ValueError, match="fill_value must be finite"):
        Imputer(fill_value=np.nan)

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
    np.testing.assert_array_equal(encoder.categories_[0], ["apple", "banana", "cherry"])

    encoded = encoder.transform([["banana"], ["cherry"], ["apple"]])
    expected = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(encoded, expected)


def test_onehot_encoder_unknown_categories_ignore():
    encoder = OneHotEncoder(handle_unknown="ignore").fit([["red"], ["blue"]])
    encoded = encoder.transform([["red"], ["green"]])
    np.testing.assert_allclose(encoded[0], [0.0, 1.0])
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
    encoded = encoder.transform(X)
    assert encoded.shape == (2, 4)


def test_standard_scaler_tiny_variance():
    # Variance of 1e-18 is non-zero and should NOT be treated as constant
    X = np.array([[1e-9], [2e-9], [3e-9]])
    scaler = StandardScaler().fit(X)
    assert scaler.var_[0] > 0.0
    assert scaler.scale_[0] < 1.0
    transformed = scaler.transform(X)
    np.testing.assert_allclose(np.var(transformed, axis=0), [1.0], atol=1e-10)


def test_standard_scaler_large_offsets_small_variation():
    offset = 1e12
    chunk1 = offset + np.array([[1.0], [2.0]])
    chunk2 = offset + np.array([[3.0], [4.0]])
    scaler = StandardScaler().partial_fit(chunk1).partial_fit(chunk2)

    np.testing.assert_allclose(scaler.mean_, [offset + 2.5])
    np.testing.assert_allclose(scaler.var_, [1.25])
    transformed = scaler.transform(offset + np.array([[2.5]]))
    np.testing.assert_allclose(transformed, [[0.0]], atol=1e-6)


def test_standard_scaler_all_nan_column():
    chunk1 = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    chunk2 = np.array([[np.nan, 3.0], [np.nan, 4.0]])
    scaler = StandardScaler().partial_fit(chunk1).partial_fit(chunk2)

    assert scaler.mean_[0] == 0.0
    assert scaler.var_[0] == 0.0
    assert scaler.scale_[0] == 1.0
    assert scaler.mean_[1] == 2.5

    transformed = scaler.transform(chunk1)
    assert np.all(np.isnan(transformed[:, 0]))
    np.testing.assert_allclose(transformed[:, 1], [-1.341641, -0.447214], atol=1e-4)


def test_imputer_mixed_and_all_nan_columns():
    chunk1 = np.array([[np.nan, 10.0, np.nan], [np.nan, np.nan, 100.0]])
    chunk2 = np.array([[np.nan, 30.0, 200.0], [np.nan, 20.0, np.nan]])

    imputer = Imputer(strategy="mean", fill_value=-1.0)
    imputer.partial_fit(chunk1).partial_fit(chunk2)

    # Col 0: all-NaN -> fallback to fill_value (-1.0)
    # Col 1: mean of 10, 30, 20 = 20.0
    # Col 2: mean of 100, 200 = 150.0
    np.testing.assert_allclose(imputer.statistics_, [-1.0, 20.0, 150.0])


def test_onehot_encoder_nan_rejected():
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match="X contains NaN"):
        encoder.fit([[1.0], [np.nan]])


def test_onehot_encoder_string_nan_supported():
    # String "nan" is a valid string category and must be handled deterministically
    X = np.array([["nan"], ["apple"], ["banana"], ["nan"]])
    encoder = OneHotEncoder().fit(X)
    assert "nan" in encoder.categories_[0]
    encoded = encoder.transform([["nan"], ["apple"]])
    assert encoded.shape == (2, 3)


def test_onehot_encoder_batch_vs_incremental_equivalence():
    chunk1 = np.array([["a"], ["b"]])
    chunk2 = np.array([["b"], ["c"]])
    full_X = np.array([["a"], ["b"], ["b"], ["c"]])

    inc_enc = OneHotEncoder().partial_fit(chunk1).partial_fit(chunk2)
    batch_enc = OneHotEncoder().fit(full_X)

    np.testing.assert_array_equal(inc_enc.categories_[0], batch_enc.categories_[0])
    np.testing.assert_array_equal(inc_enc.transform(full_X), batch_enc.transform(full_X))
