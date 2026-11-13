"""NumCompute Stream — Streaming Decision Tree and Online Analytics Framework.

Extends NumCompute for streaming classification, buffered retraining,
incremental preprocessing, online statistics, and stateful evaluation.
"""
from __future__ import annotations

from numcompute.utils import NotFittedError

from .io import CSVReader
from .metrics import (
    RollingMetrics,
    StreamingMetrics,
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc_score,
)
from .pipeline import Pipeline
from .preprocessing import Imputer, OneHotEncoder, StandardScaler
from .stats import StreamingStats, update_stats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__version__ = "0.2.0"

__all__ = [
    "DecisionTreeClassifier",
    "Pipeline",
    "StreamTrainer",
    "StandardScaler",
    "Imputer",
    "OneHotEncoder",
    "StreamingStats",
    "update_stats",
    "StreamingMetrics",
    "RollingMetrics",
    "CSVReader",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "roc_auc_score",
    "NotFittedError",
    "__version__",
]
