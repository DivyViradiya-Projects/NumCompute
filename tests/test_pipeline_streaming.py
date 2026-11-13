"""Tests for streaming Pipeline and raw buffer coordinate consistency."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute.utils import NotFittedError
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_step_validation():
    # Empty steps raises ValueError
    with pytest.raises(ValueError, match="steps must not be empty"):
        Pipeline([])

    # Duplicate names raise ValueError
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("step", StandardScaler()), ("step", DecisionTreeClassifier())])

    # Empty step name
    with pytest.raises(ValueError, match="non-empty"):
        Pipeline([("", StandardScaler()), ("clf", DecisionTreeClassifier())])

    # Intermediate step missing transform raises TypeError
    with pytest.raises(TypeError, match="must implement 'transform'"):
        Pipeline([("step1", object()), ("clf", DecisionTreeClassifier())])

    # Final step missing fit/predict raises TypeError
    with pytest.raises(TypeError, match="must implement 'fit' and 'predict'"):
        Pipeline([("scaler", StandardScaler()), ("clf", StandardScaler())])

    # Invalid buffer_size
    with pytest.raises(ValueError, match="buffer_size"):
        Pipeline([("clf", DecisionTreeClassifier())], buffer_size=0)

    with pytest.raises(ValueError, match="buffer_size"):
        Pipeline([("clf", DecisionTreeClassifier())], buffer_size=True)


def test_pipeline_batch_fit_and_predict():
    X = np.array([
        [1.0, np.nan],
        [2.0, 10.0],
        [8.0, 80.0],
        [9.0, np.nan],
    ])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imputer", Imputer()),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(criterion="gini")),
    ])

    pipe.fit(X, y)
    assert pipe.is_fitted_
    assert pipe.n_retrains_ == 1

    # Predict
    preds = pipe.predict([[1.5, 5.0], [8.5, 75.0]])
    np.testing.assert_array_equal(preds, [0, 1])
    assert pipe.score(X, y) == 1.0


def test_pipeline_streaming_partial_fit_and_retraining():
    pipe = Pipeline([
        ("imputer", Imputer()),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=10)),
    ])
    assert pipe.buffer_size == 10

    # Chunk 1 of 6 samples -> holds in buffer
    X1 = np.ones((6, 2))
    y1 = np.zeros(6)
    pipe.partial_fit(X1, y1)

    assert not pipe.is_fitted_
    assert pipe.n_buffered_samples_ == 6
    assert pipe.n_samples_seen_ == 6
    assert pipe.n_retrains_ == 0

    with pytest.raises(NotFittedError):
        pipe.predict([[1.0, 1.0]])

    # Chunk 2 of 5 samples -> total 11 >= 10 -> triggers retrain!
    X2 = np.ones((5, 2)) * 10.0
    y2 = np.ones(5)
    pipe.partial_fit(X2, y2)

    assert pipe.is_fitted_
    assert pipe.n_buffered_samples_ == 0
    assert pipe.n_samples_seen_ == 11
    assert pipe.n_retrains_ == 1

    # Predictions now functional
    preds = pipe.predict([[1.0, 1.0], [10.0, 10.0]])
    np.testing.assert_array_equal(preds, [0, 1])


def test_pipeline_committed_snapshot_prevents_scaler_drift():
    """Verify that shifts in candidate scaler mean do NOT corrupt inference on current model.

    Initial fit on [0.0] and [10.0] commits a scaler with mean 5.0.
    A non-triggering chunk with [1000.0] shifts the candidate scaler's mean to ~200,
    but prediction MUST still use the committed scaler with mean 5.0!
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=10)),
    ])

    # Initial 10 samples to train first model
    pipe.partial_fit(np.zeros((5, 1)), np.zeros(5))
    pipe.partial_fit(np.ones((5, 1)) * 10.0, np.ones(5))
    assert pipe.is_fitted_

    pred_before = pipe.predict([[0.0]])[0]
    assert pred_before == 0

    # Ingest 3 huge outlier samples that shift candidate scaler mean drastically
    pipe.partial_fit(np.ones((3, 1)) * 1000.0, np.ones(3))
    assert pipe.n_buffered_samples_ == 3  # not retrained yet!

    # Prediction on [[0.0]] MUST remain identical and uncorrupted!
    pred_after = pipe.predict([[0.0]])[0]
    assert pred_after == 0


def test_pipeline_committed_snapshot_prevents_encoder_dimension_mismatch():
    """Verify that a new category discovered in a non-triggering chunk does not break prediction."""
    pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("tree", DecisionTreeClassifier(buffer_size=6)),
    ])

    # Initial 6 samples with "cat" and "dog"
    pipe.partial_fit([["cat"]]*3 + [["dog"]]*3, [0]*3 + [1]*3)
    assert pipe.is_fitted_
    assert pipe.predict([["cat"]])[0] == 0

    # Non-triggering chunk introduces a NEW category "elephant" (2 samples < 6)
    pipe.partial_fit([["elephant"], ["dog"]], [1, 1])
    assert pipe.n_buffered_samples_ == 2

    # Prediction must NOT crash with dimension mismatch!
    assert pipe.predict([["cat"]])[0] == 0

    # Add 4 more samples to trigger retraining on new category width
    pipe.partial_fit([["elephant"]]*4, [1]*4)
    assert pipe.n_buffered_samples_ == 0
    assert pipe.n_retrains_ == 2
    # Now elephant is part of committed model!
    assert pipe.predict([["elephant"]])[0] == 1


def test_pipeline_predictions_between_retrains():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=10)),
    ])

    # Initial 10 samples to train first tree
    pipe.partial_fit(np.zeros((5, 1)), np.zeros(5))
    pipe.partial_fit(np.ones((5, 1)) * 10.0, np.ones(5))
    assert pipe.is_fitted_
    assert pipe.n_retrains_ == 1

    # Ingest 3 new samples with opposite labels (pending = 3 < 10)
    pipe.partial_fit(np.zeros((3, 1)), np.ones(3))
    assert pipe.n_buffered_samples_ == 3
    assert pipe.n_retrains_ == 1
    # Prediction must still be served by the existing model
    assert pipe.predict([[0.0]])[0] == 0


def test_pipeline_flush():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=100)),
    ])
    pipe.partial_fit([[0.0], [1.0]], [0, 0])
    pipe.partial_fit([[10.0], [11.0]], [1, 1])

    assert not pipe.is_fitted_
    assert pipe.n_buffered_samples_ == 4

    pipe.flush()
    assert pipe.is_fitted_
    assert pipe.n_buffered_samples_ == 0
    assert pipe.n_retrains_ == 1
    assert pipe.predict([[0.5]])[0] == 0
    assert pipe.predict([[10.5]])[0] == 1
