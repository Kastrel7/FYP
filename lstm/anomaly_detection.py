import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

WINDOW_SIZE = 30

model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("feature_order.pkl")

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

def build_sequences(entries):
    flattened = [flatten_entry(e) for e in entries]

    df = pd.DataFrame(flattened, columns=FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaled = scaler.transform(df)

    X, actual, timestamps = [], [], []

    for i in range(len(entries) - WINDOW_SIZE):
        X.append(scaled[i:i+WINDOW_SIZE])
        actual.append(scaled[i+WINDOW_SIZE])

        ts = entries[i+WINDOW_SIZE].get("timestamp_iso") or entries[i+WINDOW_SIZE].get("timestamp")
        timestamps.append(ts)

    return np.array(X), np.array(actual), timestamps

def compute_scores(X, actual):
    preds = model.predict(X, verbose=0)
    errors = np.mean(np.abs(preds - actual), axis=1)
    return errors[~np.isnan(errors)]


def compute_threshold(scores):
    scores = scores[~np.isnan(scores)]
    return scores.mean() + 3 * scores.std()

def write_anomalies(scores, timestamps, threshold, outfile="anomalies.log"):
    with open(outfile, "w") as f:
        for i, score in enumerate(scores):
            if score > threshold:
                ts = timestamps[i]
                f.write(f"{ts} | score={score:.6f} -> ANOMALY\n")

        f.write("\nFinished anomaly scan.\n")

    print(f"Anomalies written to {outfile}")

def plot_scores(scores, threshold, outfile="anomaly_scores.png"):
    plt.figure(figsize=(14, 5))
    plt.plot(scores, label="Anomaly Score")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.title("Anomaly Scores Over Time")
    plt.xlabel("Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Plot saved to {outfile}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 anomaly_log_and_plot.py <file.jsonl>")
        exit(1)

    path = sys.argv[1]

    entries = load_jsonl(path)
    X, actual, timestamps = build_sequences(entries)

    scores = compute_scores(X, actual)
    threshold = compute_threshold(scores)

    print(f"Threshold: {threshold:.6f}")

    write_anomalies(scores, timestamps, threshold)
    plot_scores(scores, threshold)
