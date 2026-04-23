import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

WINDOW_SIZE = 30
PREDICTION_HORIZON = 12  # 12 steps x 5 seconds = 60 seconds ahead

model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("feature_order.pkl")
threshold_info = joblib.load("threshold_info.pkl")
THRESHOLD = threshold_info["threshold"]

print(f"Loaded threshold: {THRESHOLD:.6f}")

def flatten_entry(entry):
    row = {}

    for key, value in entry.get("host", {}).items():
        row[f"host_{key}"] = value

    for cname, cdata in entry.get("containers", {}).items():
        for mkey, mval in cdata.get("metrics", {}).items():
            row[f"{cname}_{mkey}"] = mval

    for feat in FEATURES:
        if feat not in row:
            row[feat] = 0

    return [row[f] for f in FEATURES]

def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass
    return entries

def predict_ahead_single(window, horizon):
    """
    Recursively predict horizon steps ahead for a single window.
    Best for live use where you score one window at a time.
    """
    current_window = window.copy()

    for _ in range(horizon):
        x = current_window[np.newaxis, :, :]
        next_step = model.predict(x, verbose=0)[0]
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1] = next_step

    return next_step

def predict_ahead_batch(windows, horizon):
    """
    Recursively predict horizon steps ahead for all windows at once.
    Best for pre-collected files where you want speed.
    """
    current_windows = np.array(windows).copy()

    for _ in range(horizon):
        next_steps = model.predict(current_windows, verbose=0)
        current_windows = np.roll(current_windows, -1, axis=1)
        current_windows[:, -1, :] = next_steps

    return next_steps

def build_sequences(entries):
    flattened = [flatten_entry(e) for e in entries]

    df = pd.DataFrame(flattened, columns=FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaled = scaler.transform(df)

    windows = []
    timestamps = []

    for i in range(len(scaled) - WINDOW_SIZE):
        windows.append(scaled[i:i+WINDOW_SIZE])
        ts = entries[i+WINDOW_SIZE].get("timestamp_iso") or entries[i+WINDOW_SIZE].get("timestamp")
        timestamps.append(ts)

    return windows, timestamps

def compute_scores_single(windows):
    scores = []

    for i, window in enumerate(windows):
        prediction = predict_ahead_single(window, PREDICTION_HORIZON)
        score = float(np.mean(np.abs(prediction)))
        scores.append(score)

        if (i + 1) % 50 == 0:
            print(f"Scored {i + 1}/{len(windows)} windows...")

    return np.array(scores)

def compute_scores_batch(windows):
    print("Running batch prediction...")
    predictions = predict_ahead_batch(windows, PREDICTION_HORIZON)
    scores = np.mean(np.abs(predictions), axis=1)
    print(f"Scored {len(windows)} windows in batch mode")
    return scores

def write_anomalies(scores, timestamps, outfile="anomalies.log"):
    anomaly_count = 0

    with open(outfile, "w") as f:
        for i, score in enumerate(scores):
            if score > THRESHOLD:
                ts = timestamps[i]
                f.write(f"{ts} | score={score:.6f} | predicted_anomaly_in_60s -> ANOMALY\n")
                anomaly_count += 1

        f.write(f"\nFinished anomaly scan. {anomaly_count} anomalies detected.\n")

    print(f"Anomalies written to {outfile} ({anomaly_count} detected)")

def plot_scores(scores, outfile="anomaly_scores.png"):
    plt.figure(figsize=(14, 5))
    plt.plot(scores, label="Anomaly Score (1 min ahead)")
    plt.axhline(THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD:.4f})")
    plt.title("Predicted Anomaly Scores Over Time (60 Second Lookahead)")
    plt.xlabel("Window Index")
    plt.ylabel("Predicted Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Plot saved to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection against a JSONL dataset")
    parser.add_argument("-i", "--input", required=True, help="Path to the JSONL dataset file")
    parser.add_argument("-b", "--batch", action="store_true", help="Use batch mode for faster scoring of large files")
    args = parser.parse_args()

    entries = load_jsonl(args.input)
    print(f"Loaded {len(entries)} entries from {args.input}")

    windows, timestamps = build_sequences(entries)
    print(f"Built {len(windows)} windows")
    print(f"Mode: {'batch' if args.batch else 'single'}")

    if args.batch:
        scores = compute_scores_batch(windows)
    else:
        scores = compute_scores_single(windows)

    print(f"Threshold: {THRESHOLD:.6f}")
    print(f"Max score: {scores.max():.6f}")
    print(f"Mean score: {scores.mean():.6f}")

    write_anomalies(scores, timestamps)
    plot_scores(scores)
