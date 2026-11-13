"""End-to-End Streaming Integration Scenarios.

Implements the 6 required integration scenarios from Assignment 2.2 specifications:
1. CSVReader -> Imputer -> StandardScaler -> Pipeline -> StreamTrainer
2. StandardScaler -> DecisionTreeClassifier streaming pipeline
3. OneHotEncoder -> estimator pipeline with new categories discovered in later chunks
4. Multiple chunks with buffer sizes smaller than, equal to, and larger than chunk size
5. Prequential evaluation over several chunks, verifying test-then-train ordering
6. Failed retraining followed by a successful retry
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from numcompute_stream.io import CSVReader
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


# =====================================================================
# Scenario 1: CSVReader -> Preprocessing -> Pipeline -> StreamTrainer
# =====================================================================
def test_scenario_1_csvreader_preprocessing_pipeline_streamtrainer(tmp_path):
    """Full streaming loop: reading from disk, imputing, scaling, and training."""
    csv_file = tmp_path / "stream_data.csv"
    rng = np.random.RandomState(42)

    # 40 samples, 2 features, binary target
    lines = ["f1,f2,target\n"]
    for i in range(40):
        v1 = f"{rng.normal(0, 1):.4f}" if rng.rand() > 0.1 else "NaN"
        v2 = f"{rng.normal(2, 1):.4f}"
        lbl = 1 if (v1 != "NaN" and float(v1) > 0) else 0
        lines.append(f"{v1},{v2},{lbl}\n")
    csv_file.write_text("".join(lines))

    reader = CSVReader(csv_file, target_column="target")
    pipe = Pipeline([
        ("imputer", Imputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=15, random_state=42)),
    ], buffer_size=15)
    trainer = StreamTrainer(pipe)

    # Ingest in chunks of 10 samples
    chunk_count = 0
    for X_chunk, y_chunk in reader.iter_chunks(chunk_size=10):
        log = trainer.fit_chunk(X_chunk, y_chunk)
        chunk_count += 1
        assert "chunk_index" in log
        assert "memory_bytes" in log

    assert chunk_count == 4
    assert trainer.n_samples_seen_ == 40
    assert trainer.model.is_fitted_
    assert trainer.model.n_retrains_ >= 2

    # Query fitted pipeline
    preds = pipe.predict([[1.5, 2.0], [-1.5, 2.0]])
    assert len(preds) == 2


# =====================================================================
# Scenario 2: StandardScaler -> DecisionTreeClassifier Streaming Pipeline
# =====================================================================
def test_scenario_2_scaler_tree_pipeline():
    """Verify that scaling features maintains separation and fits tree accurately."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=20, max_depth=3, random_state=42)),
    ], buffer_size=20)

    rng = np.random.RandomState(123)
    # Balanced classes distributed across stream
    n_samples = 40
    c0 = rng.normal(1000, 10, size=(n_samples // 2, 2))
    c1 = rng.normal(2000, 10, size=(n_samples // 2, 2))
    X = np.empty((n_samples, 2))
    y = np.empty(n_samples, dtype=int)
    X[0::2] = c0
    y[0::2] = 0
    X[1::2] = c1
    y[1::2] = 1

    # Ingest in chunks of 10 samples
    pipe.partial_fit(X[:10], y[:10])
    assert not pipe.is_fitted_
    pipe.partial_fit(X[10:20], y[10:20])
    assert pipe.is_fitted_
    assert pipe.n_retrains_ == 1

    pipe.partial_fit(X[20:30], y[20:30])
    pipe.partial_fit(X[30:40], y[30:40])
    assert pipe.n_retrains_ == 2

    test_samples = np.array([[1000.0, 1000.0], [2000.0, 2000.0]])
    preds = pipe.predict(test_samples)
    np.testing.assert_array_equal(preds, [0, 1])


# =====================================================================
# Scenario 3: OneHotEncoder -> Estimator Pipeline with Emerging Categories
# =====================================================================
def test_scenario_3_encoder_emerging_categories_pipeline():
    """Verify that new categories in later chunks don't break snapshot inference."""
    pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("tree", DecisionTreeClassifier(buffer_size=10, random_state=42)),
    ])

    # Initial chunk: only 'cat' and 'dog'
    X1 = np.array([["cat"]]*5 + [["dog"]]*5)
    y1 = np.array([0]*5 + [1]*5)
    pipe.partial_fit(X1, y1)
    assert pipe.is_fitted_
    assert pipe.predict([["cat"]])[0] == 0
    assert pipe.predict([["dog"]])[0] == 1

    # Chunk 2 introduces 'bird' (non-triggering chunk of 4 samples)
    X2 = np.array([["bird"], ["bird"], ["cat"], ["dog"]])
    y2 = np.array([1, 1, 0, 1])
    pipe.partial_fit(X2, y2)
    assert pipe.n_buffered_samples_ == 4
    # Old snapshot still serves predictions without dimension crash
    assert pipe.predict([["cat"]])[0] == 0

    # Chunk 3 has 6 samples -> triggers retrain on unified 3 categories
    X3 = np.array([["bird"]]*4 + [["cat"]]*2)
    y3 = np.array([1]*4 + [0]*2)
    pipe.partial_fit(X3, y3)
    assert pipe.is_fitted_
    assert pipe.n_retrains_ == 2
    assert pipe.n_buffered_samples_ == 0
    # Now 'bird' is recognized as class 1
    assert pipe.predict([["bird"]])[0] == 1


# =====================================================================
# Scenario 4: Buffer Sizes: Smaller than, Equal to, and Larger than Chunk
# =====================================================================
def test_scenario_4_buffer_size_relationships():
    X_chunk = np.ones((10, 2))
    y_chunk = np.array([0]*5 + [1]*5)

    # 4a. Buffer size SMALLER than chunk (buffer=5, chunk=10)
    tree_small = DecisionTreeClassifier(buffer_size=5)
    tree_small.partial_fit(X_chunk, y_chunk)
    assert tree_small.is_fitted_
    assert tree_small.n_retrains_ == 1
    assert tree_small.n_buffered_samples_ == 0

    # 4b. Buffer size EQUAL to chunk (buffer=10, chunk=10)
    tree_equal = DecisionTreeClassifier(buffer_size=10)
    tree_equal.partial_fit(X_chunk, y_chunk)
    assert tree_equal.is_fitted_
    assert tree_equal.n_retrains_ == 1
    assert tree_equal.n_buffered_samples_ == 0

    # 4c. Buffer size LARGER than chunk (buffer=25, chunk=10)
    tree_large = DecisionTreeClassifier(buffer_size=25)
    tree_large.partial_fit(X_chunk, y_chunk)
    assert not tree_large.is_fitted_
    assert tree_large.n_buffered_samples_ == 10

    tree_large.partial_fit(X_chunk, y_chunk)
    assert not tree_large.is_fitted_
    assert tree_large.n_buffered_samples_ == 20

    tree_large.partial_fit(X_chunk, y_chunk)
    assert tree_large.is_fitted_
    assert tree_large.n_retrains_ == 1
    assert tree_large.n_buffered_samples_ == 0
    assert tree_large.n_samples_seen_ == 30


# =====================================================================
# Scenario 5: Strict Prequential Evaluation (Test-Then-Train)
# =====================================================================
def test_scenario_5_prequential_test_then_train_ordering():
    """Verify that chunk evaluation strictly uses the model as it existed BEFORE training."""
    tree = DecisionTreeClassifier(buffer_size=4, random_state=42)
    trainer = StreamTrainer(tree)

    # Initial chunk 1 (4 samples) fits the initial model on classes 0 and 1
    c1_X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
    c1_y = np.array([0, 0, 1, 1])
    log1 = trainer.fit_chunk(c1_X, c1_y)
    # Chunk 1 had no fitted model to test against
    assert log1["chunk_accuracy"] is None
    assert log1["retrained"]
    assert log1["retrain_count"] == 1

    # Chunk 2 has samples that match the trained model (0->0, 10->1)
    c2_X = np.array([[0.0, 0.0], [10.0, 10.0]])
    c2_y = np.array([0, 1])
    log2 = trainer.fit_chunk(c2_X, c2_y)
    # Prequential accuracy is evaluated BEFORE chunk 2 modifies training state -> 100%
    assert log2["chunk_accuracy"] == 1.0
    assert log2["cumulative_accuracy"] == 1.0

    # Chunk 3 has samples with intentionally flipped labels
    c3_X = np.array([[0.0, 0.0], [10.0, 10.0]])
    c3_y = np.array([1, 0])
    log3 = trainer.fit_chunk(c3_X, c3_y)
    # Evaluated against existing model -> 0.0 accuracy!
    assert log3["chunk_accuracy"] == 0.0
    # Cumulative evaluated = 2 (chunk 2) + 2 (chunk 3) = 4. Correct = 2 + 0 = 2.
    assert log3["cumulative_accuracy"] == 0.5


# =====================================================================
# Scenario 6: Failed Retraining Followed by Successful Retry
# =====================================================================
def test_scenario_6_failed_retraining_and_successful_retry():
    """Verify that a transient retraining failure preserves the model and buffer for retry."""
    tree = DecisionTreeClassifier(buffer_size=4)
    # Fit initial working model
    tree.partial_fit([[0.0], [0.0], [1.0], [1.0]], [0, 0, 1, 1])
    assert tree.is_fitted_
    assert tree.n_retrains_ == 1
    assert tree.predict([[0.0]])[0] == 0
    assert tree.predict([[1.0]])[0] == 1

    # Simulate transient hardware or memory failure during next retrain
    real_build = tree._build_tree

    def transient_failure(*args, **kwargs):
        raise RuntimeError("Transient CUDA/Memory fault during node split")

    tree._build_tree = transient_failure

    # Attempt to ingest chunk that triggers retraining
    with pytest.raises(RuntimeError, match="Transient"):
        tree.partial_fit([[2.0], [2.0], [2.0], [2.0]], [2, 2, 2, 2])

    # Old model MUST remain intact and usable!
    assert tree.is_fitted_
    assert tree.predict([[0.0]])[0] == 0
    assert tree.predict([[1.0]])[0] == 1

    # Now recover from transient failure
    tree._build_tree = real_build

    # Retry retraining with flush or partial_fit
    tree.flush()
    assert tree.is_fitted_
    assert tree.n_retrains_ == 2
    assert tree.n_buffered_samples_ == 0
    assert tree.predict([[2.0]])[0] == 2
