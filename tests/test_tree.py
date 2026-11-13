"""Tests for DecisionTreeClassifier and buffered retraining semantics."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute.utils import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier, _compute_impurity


# =====================================================================
# Impurity Function Tests
# =====================================================================
def test_compute_impurity_gini():
    # Pure node: Gini = 0.0
    assert _compute_impurity(np.array([10, 0]), 10, "gini") == 0.0
    # Balanced binary: Gini = 1 - (0.5^2 + 0.5^2) = 0.5
    assert _compute_impurity(np.array([5, 5]), 10, "gini") == 0.5
    # 3-class balanced: Gini = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
    assert np.isclose(_compute_impurity(np.array([4, 4, 4]), 12, "gini"), 2.0 / 3.0)


def test_compute_impurity_entropy():
    # Pure node: Entropy = 0.0
    assert _compute_impurity(np.array([10, 0]), 10, "entropy") == 0.0
    # Balanced binary: Entropy = - (0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
    assert _compute_impurity(np.array([5, 5]), 10, "entropy") == 1.0
    # 4-class balanced: Entropy = log2(4) = 2.0
    assert np.isclose(_compute_impurity(np.array([3, 3, 3, 3]), 12, "entropy"), 2.0)


def test_compute_impurity_invalid_criterion():
    with pytest.raises(ValueError, match="criterion must be 'gini' or 'entropy'"):
        _compute_impurity(np.array([5, 5]), 10, "misclassification_error")


# =====================================================================
# Basic Decision Tree Fitting and Prediction
# =====================================================================
def test_tree_known_split_1d():
    # Simple linearly separable 1D dataset: x <= 5 -> class 0, x > 5 -> class 1
    X = np.array([[1.0], [2.0], [3.0], [4.0], [7.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    tree = DecisionTreeClassifier(criterion="gini")
    tree.fit(X, y)

    assert tree.is_fitted_
    assert tree.get_depth() == 1
    assert tree.get_n_leaves() == 2

    # Verify split threshold is between 4 and 7 (e.g. 5.5)
    assert tree.tree_ is not None
    assert tree.tree_.feature == 0
    assert np.isclose(tree.tree_.threshold, 5.5)

    # Predictions
    preds = tree.predict(np.array([[2.5], [6.0], [12.0]]))
    np.testing.assert_array_equal(preds, [0, 1, 1])

    # Probabilities
    proba = tree.predict_proba(np.array([[2.5], [6.0]]))
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba[0], [1.0, 0.0])
    np.testing.assert_allclose(proba[1], [0.0, 1.0])


def test_tree_entropy_criterion():
    X = np.array([[0.5], [1.5], [2.5], [3.5]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(X, y)
    assert tree.score(X, y) == 1.0


def test_tree_multiclass():
    # 3 classes: A, B, C along 2 features
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [5.0, 5.0],
        [5.5, 5.5],
        [10.0, 10.0],
        [10.5, 10.5],
    ])
    y = np.array(["class_A", "class_A", "class_B", "class_B", "class_C", "class_C"])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    assert tree.n_classes_ == 3
    np.testing.assert_array_equal(tree.classes_, ["class_A", "class_B", "class_C"])
    assert tree.score(X, y) == 1.0

    test_queries = np.array([[0.2, 0.1], [5.2, 5.1], [10.2, 10.1]])
    np.testing.assert_array_equal(tree.predict(test_queries), ["class_A", "class_B", "class_C"])


def test_tree_pure_data_one_class():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    assert tree.get_depth() == 0
    assert tree.get_n_leaves() == 1
    np.testing.assert_array_equal(tree.predict(X), [1, 1, 1])


def test_tree_max_depth_constraint():
    # Requires multiple splits if unconstrained
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    assert tree.get_depth() <= 1


def test_tree_min_samples_split_constraint():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    # min_samples_split=10 prevents any splitting on 4 samples
    tree = DecisionTreeClassifier(min_samples_split=10)
    tree.fit(X, y)
    assert tree.get_depth() == 0
    assert tree.get_n_leaves() == 1


def test_tree_constant_features_no_split():
    # Identical feature values across different classes
    X = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    assert tree.get_depth() == 0
    assert tree.get_n_leaves() == 1
    # Class tie resolved deterministically to class 0 (lowest class index)
    assert tree.predict([[5.0, 5.0]])[0] == 0


def test_tree_deterministic_tie_breaking():
    # Two features with identical split quality
    X = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0],
    ])
    y = np.array([0, 0, 1, 1])

    # Feature 0 should be chosen deterministically over feature 1
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    assert tree.tree_ is not None
    assert tree.tree_.feature == 0


def test_tree_max_features_options():
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # None
    tree_all = DecisionTreeClassifier(max_features=None, random_state=42).fit(X, y)
    assert tree_all.is_fitted_

    # int
    tree_int = DecisionTreeClassifier(max_features=2, random_state=42).fit(X, y)
    assert tree_int.is_fitted_

    # float
    tree_float = DecisionTreeClassifier(max_features=0.5, random_state=42).fit(X, y)
    assert tree_float.is_fitted_

    # "sqrt"
    tree_sqrt = DecisionTreeClassifier(max_features="sqrt", random_state=42).fit(X, y)
    assert tree_sqrt.is_fitted_

    # "log2"
    tree_log2 = DecisionTreeClassifier(max_features="log2", random_state=42).fit(X, y)
    assert tree_log2.is_fitted_


# =====================================================================
# Error & Edge Case Handling Tests
# =====================================================================
def test_tree_predict_before_fit_raises():
    tree = DecisionTreeClassifier()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        tree.predict(np.array([[1.0, 2.0]]))

    with pytest.raises(NotFittedError, match="not fitted yet"):
        tree.predict_proba(np.array([[1.0, 2.0]]))

    with pytest.raises(NotFittedError, match="not fitted yet"):
        tree.get_depth()

    with pytest.raises(NotFittedError, match="not fitted yet"):
        tree.get_n_leaves()


def test_tree_rejects_nan_and_inf():
    tree = DecisionTreeClassifier()
    X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="NaN or Inf"):
        tree.fit(X_nan, y)

    X_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        tree.fit(X_inf, y)


def test_tree_invalid_shapes_and_types():
    tree = DecisionTreeClassifier()
    # 1D X
    with pytest.raises(ValueError):
        tree.fit([1.0, 2.0, 3.0], [0, 1, 0])

    # Mismatched lengths
    with pytest.raises(ValueError, match="must match"):
        tree.fit([[1.0], [2.0]], [0, 1, 0])

    # Non-numeric X
    with pytest.raises(TypeError, match="numeric"):
        tree.fit([["a", "b"], ["c", "d"]], [0, 1])

    # Empty inputs
    with pytest.raises(ValueError):
        tree.fit(np.zeros((0, 2)), np.zeros(0))


def test_tree_feature_count_mismatch_at_predict():
    tree = DecisionTreeClassifier()
    tree.fit([[1.0, 2.0], [3.0, 4.0]], [0, 1])
    with pytest.raises(ValueError, match="fitted with 2 features"):
        tree.predict([[1.0, 2.0, 3.0]])


def test_tree_invalid_hyperparameters():
    with pytest.raises(ValueError, match="max_depth"):
        DecisionTreeClassifier(max_depth=-1)

    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTreeClassifier(min_samples_split=1)

    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="unknown")

    with pytest.raises(ValueError, match="buffer_size"):
        DecisionTreeClassifier(buffer_size=0)

    with pytest.raises(ValueError, match="max_features"):
        DecisionTreeClassifier(max_features="invalid")

    with pytest.raises(TypeError, match="max_features"):
        DecisionTreeClassifier(max_features=[1, 2])


# =====================================================================
# Buffered Retraining Streaming Semantics Tests
# =====================================================================
def test_buffered_retraining_below_threshold():
    # buffer_size = 10, feed chunk of 8 -> buffer holds 8, not fitted yet
    tree = DecisionTreeClassifier(buffer_size=10)
    X_chunk = np.ones((8, 2))
    y_chunk = np.zeros(8)

    tree.partial_fit(X_chunk, y_chunk)

    assert not tree.is_fitted_
    assert tree.n_buffered_samples_ == 8
    assert tree.n_samples_seen_ == 8
    assert tree.n_retrains_ == 0

    with pytest.raises(NotFittedError):
        tree.predict([[1.0, 1.0]])


def test_buffered_retraining_exact_threshold():
    # buffer_size = 10, feed chunk of 10 -> triggers retrain, clears buffer
    tree = DecisionTreeClassifier(buffer_size=10)
    X_chunk = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    y_chunk = np.array([0]*5 + [1]*5)

    tree.partial_fit(X_chunk, y_chunk)

    assert tree.is_fitted_
    assert tree.n_buffered_samples_ == 0
    assert tree.n_samples_seen_ == 10
    assert tree.n_retrains_ == 1

    # Model can now predict
    preds = tree.predict(np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_array_equal(preds, [0, 1])


def test_buffered_retraining_overshoot():
    # buffer_size = 10; first chunk = 8 (pending=8); second chunk = 5 (pending=13 >= 10)
    # Retrains on all 13 samples, then buffer is cleared to 0
    tree = DecisionTreeClassifier(buffer_size=10)

    X1 = np.zeros((8, 1))
    y1 = np.zeros(8)
    tree.partial_fit(X1, y1)
    assert not tree.is_fitted_
    assert tree.n_buffered_samples_ == 8

    X2 = np.ones((5, 1)) * 10.0
    y2 = np.ones(5)
    tree.partial_fit(X2, y2)

    assert tree.is_fitted_
    assert tree.n_buffered_samples_ == 0
    assert tree.n_samples_seen_ == 13
    assert tree.n_retrains_ == 1

    # Check predictions: 0.0 maps to 0, 10.0 maps to 1
    preds = tree.predict([[0.0], [10.0]])
    np.testing.assert_array_equal(preds, [0, 1])


def test_prediction_served_between_retrains():
    # Once fitted, subsequent chunks that do not trigger retrain leave the previous model serving predictions
    tree = DecisionTreeClassifier(buffer_size=10)

    # Initial 10 samples to fit first model
    X_init = np.array([[1.0]]*5 + [[9.0]]*5)
    y_init = np.array([0]*5 + [1]*5)
    tree.partial_fit(X_init, y_init)

    assert tree.is_fitted_
    assert tree.n_retrains_ == 1
    assert tree.predict([[1.0]])[0] == 0
    assert tree.predict([[9.0]])[0] == 1

    # Add 4 new conflicting samples (pending = 4 < 10)
    # Even if they have opposite labels, tree must NOT change yet!
    X_new = np.array([[1.0]]*4)
    y_new = np.array([1]*4)
    tree.partial_fit(X_new, y_new)

    assert tree.is_fitted_
    assert tree.n_buffered_samples_ == 4
    assert tree.n_retrains_ == 1
    # Prediction MUST still use previous model (predicts 0 for 1.0)
    assert tree.predict([[1.0]])[0] == 0


def test_multiple_retraining_cycles():
    # buffer_size = 5. Feed chunks of 2, 2, 2 (triggers retrain on 6), then 5 (triggers retrain on 5)
    tree = DecisionTreeClassifier(buffer_size=5)

    tree.partial_fit(np.zeros((2, 1)), np.zeros(2))
    assert tree.n_retrains_ == 0 and tree.n_buffered_samples_ == 2

    tree.partial_fit(np.zeros((2, 1)), np.zeros(2))
    assert tree.n_retrains_ == 0 and tree.n_buffered_samples_ == 4

    tree.partial_fit(np.ones((2, 1)), np.ones(2))
    # Pending was 4 + 2 = 6 >= 5 -> retrain #1!
    assert tree.n_retrains_ == 1 and tree.n_buffered_samples_ == 0

    tree.partial_fit(np.ones((5, 1)), np.ones(5))
    # Pending is 5 >= 5 -> retrain #2!
    assert tree.n_retrains_ == 2 and tree.n_buffered_samples_ == 0
    assert tree.n_samples_seen_ == 11


def test_tree_flush_method():
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1])

    assert not tree.is_fitted_
    assert tree.n_buffered_samples_ == 4

    # Flush forces retrain on pending 4 samples
    tree.flush()
    assert tree.is_fitted_
    assert tree.n_buffered_samples_ == 0
    assert tree.n_retrains_ == 1
    np.testing.assert_array_equal(tree.predict([[1.5], [8.5]]), [0, 1])

