"""Tests for StreamTrainer and prequential evaluation protocol."""
from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def test_stream_trainer_prequential_workflow():
    tree = DecisionTreeClassifier(buffer_size=10)
    trainer = StreamTrainer(tree)

    # Chunk 1: 5 samples (pending=5 < 10) -> warmup
    c1_X = np.zeros((5, 2))
    c1_y = np.zeros(5)

    log1 = trainer.fit_chunk(c1_X, c1_y)
    assert log1["chunk_index"] == 0
    assert log1["n_samples"] == 5
    assert log1["cumulative_samples"] == 5
    assert not log1["retrained"]
    assert log1["chunk_accuracy"] is None
    assert log1["cumulative_accuracy"] is None
    assert log1["fit_time"] >= 0.0
    assert log1["memory_bytes"] > 0

    # Chunk 2: 5 samples -> total 10 >= 10 -> triggers retrain #1
    c2_X = np.ones((5, 2))
    c2_y = np.ones(5)

    log2 = trainer.fit_chunk(c2_X, c2_y)
    assert log2["chunk_index"] == 1
    assert log2["cumulative_samples"] == 10
    assert log2["retrained"]
    assert log2["retrain_count"] == 1
    assert log2["chunk_accuracy"] is None

    # Chunk 3: 4 samples tested on model fitted after chunk 2
    c3_X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    c3_y = np.array([0, 0, 1, 1])

    log3 = trainer.fit_chunk(c3_X, c3_y)
    assert log3["chunk_index"] == 2
    assert not log3["retrained"]
    assert log3["chunk_accuracy"] == 1.0
    assert log3["cumulative_accuracy"] == 1.0
    assert log3["cumulative_error"] == 0.0


def test_stream_trainer_with_pipeline_and_log_immutability():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=6)),
    ])
    trainer = StreamTrainer(pipe)

    trainer.fit_chunk([[0.0], [0.0]], [0, 0])
    trainer.fit_chunk([[10.0], [10.0]], [1, 1])
    c3 = trainer.fit_chunk([[10.0], [10.0]], [1, 1])

    assert c3["retrained"]
    assert c3["retrain_count"] == 1

    logs = trainer.get_logs()
    assert len(logs) == 3
    # Mutating returned list must not affect trainer's internal logs
    logs[0]["chunk_index"] = 999
    assert trainer.get_logs()[0]["chunk_index"] == 0
