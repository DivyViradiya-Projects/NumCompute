"""Live demo for the video: stream a CSV, log every chunk, predict, plot.

Run from the repository root:
    python live_demo.py            # opens plot windows (use for the recording)
    python live_demo.py --no-show  # only saves PNGs (use to test)
"""
import sys

import numpy as np
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numcompute_stream import (
    CSVReader, Imputer, StandardScaler, DecisionTreeClassifier, Pipeline,
    StreamTrainer, compare_models, plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

CSV = "demo/stream_demo_dataset.csv"
CHUNK = 75
N_TRAIN_CHUNKS = 14          # chunks 0-13 are streamed; chunks 14-15 are held out


def build(buffer_size):
    pipe = Pipeline([
        ("imputer", Imputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=buffer_size, max_depth=5, random_state=42)),
    ], buffer_size=buffer_size)
    return StreamTrainer(pipe)


def stream(buffer_size, verbose):
    trainer = build(buffer_size)
    held_X, held_y = [], []
    reader = CSVReader(CSV, target_column="target")
    for i, (X, y) in enumerate(reader.iter_chunks(chunk_size=CHUNK)):
        if i >= N_TRAIN_CHUNKS:
            held_X.append(X)
            held_y.append(y)
            continue
        log = trainer.fit_chunk(X, y)
        if verbose:
            acc = log["chunk_accuracy"]
            cum = log["cumulative_accuracy"]
            print(f"chunk {log['chunk_index']:>2} | seen {log['cumulative_samples']:>4} | "
                  f"chunk acc {'warm-up' if acc is None else f'{acc:.3f}':>7} | "
                  f"cumulative {'  --  ' if cum is None else f'{cum:.3f}'} | "
                  f"buffered {log['n_buffered_samples']:>3} | "
                  f"{'RETRAINED' if log['retrained'] else '':<9} | "
                  f"fit {log['fit_time'] * 1000:6.2f} ms")
    return trainer, np.vstack(held_X), np.concatenate(held_y)


print("=== Streaming with buffer_size=150 (prequential: test, then train) ===")
trainer, X_test, y_test = stream(150, verbose=True)
logs = trainer.get_logs()
print(f"\nRetrains: {logs[-1]['retrain_count']}   "
      f"Final cumulative accuracy: {logs[-1]['cumulative_accuracy']:.4f}")

print("\n=== Predictions on 150 held-out rows (never trained on) ===")
y_pred = trainer.model.predict(X_test)
print("first 15 true     :", y_test[:15].astype(int))
print("first 15 predicted:", y_pred[:15].astype(int))
print(f"held-out accuracy : {np.mean(y_pred == y_test):.4f}")

print("\n=== Plots ===")
acc_150 = [l["chunk_accuracy"] for l in logs]
retrains = [l["chunk_index"] for l in logs if l["retrained"]]
plot_metric_over_time(acc_150, title="Chunk accuracy over the stream (buffer 150)",
                      retrain_points=retrains, save_path="live_accuracy.png")
tr_50, _, _ = stream(50, verbose=False)
tr_250, _, _ = stream(250, verbose=False)
compare_models([l["cumulative_accuracy"] for l in tr_50.get_logs()],
               [l["cumulative_accuracy"] for l in tr_250.get_logs()],
               labels=("buffer 50", "buffer 250"),
               title="Cumulative accuracy: buffer 50 vs 250", ylabel="Cumulative accuracy",
               save_path="live_compare.png")
plot_predictions_vs_ground_truth(y_test[:60], y_pred[:60],
                                 title="Held-out predictions vs ground truth (first 60)",
                                 save_path="live_predictions.png")
print("saved live_accuracy.png, live_compare.png, live_predictions.png")
if "--no-show" not in sys.argv:
    plt.show()
