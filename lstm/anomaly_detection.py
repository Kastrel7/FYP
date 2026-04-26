import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

WINDOW_SIZE = 30
PREDICTION_HORIZON = 12

THRESHOLD_CPU     = 75
THRESHOLD_MEMORY  = 70
THRESHOLD_NET_RX  = 10000
THRESHOLD_NET_TX  = 50000

model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("feature_order.pkl")
threshold_info = joblib.load("threshold_info.pkl")
LSTM_THRESHOLD = threshold_info["threshold"]

print(f"Loaded LSTM threshold: {LSTM_THRESHOLD:.6f}")

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

def check_thresholds(entry):
    breaches = []
    host = entry.get("host", {})

    cpu = host.get("cpu_percent")
    if cpu is not None and cpu > THRESHOLD_CPU:
        breaches.append(f"CPU({cpu}%)")

    memory = host.get("memory_usage")
    if memory is not None and memory > THRESHOLD_MEMORY:
        breaches.append(f"MEM({memory}%)")

    net_rx = host.get("net_rx")
    if net_rx is not None and net_rx > THRESHOLD_NET_RX:
        breaches.append(f"NET_RX({net_rx}B/s)")

    net_tx = host.get("net_tx")
    if net_tx is not None and net_tx > THRESHOLD_NET_TX:
        breaches.append(f"NET_TX({net_tx}B/s)")

    return breaches

def predict_ahead_single(window, horizon):
    current_window = window.copy()

    for _ in range(horizon):
        x = current_window[np.newaxis, :, :]
        next_step = model.predict(x, verbose=0)[0]
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1] = next_step

    return next_step

def predict_ahead_batch(windows, horizon):
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

def score_window(window):
    prediction = predict_ahead_single(window, PREDICTION_HORIZON)
    return float(np.mean(np.abs(prediction)))

def get_alert_status(score, breaches):
    lstm_alert = score > LSTM_THRESHOLD
    threshold_alert = len(breaches) > 0

    if lstm_alert and threshold_alert:
        return "HYBRID_ALERT"
    elif lstm_alert:
        return "LSTM_ANOMALY"
    elif threshold_alert:
        return "THRESHOLD_ALERT"
    return None

def format_alert_line(ts, score, breaches, status):
    breach_str = ", ".join(breaches) if breaches else "CLEAR"
    return f"{ts} | lstm_score={score:.6f} | threshold={breach_str} | -> {status}"

def compute_scores_single(windows):
    scores = []

    for i, window in enumerate(windows):
        score = score_window(window)
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

def write_anomalies(scores, timestamps, entries, outfile="anomalies.log"):
    lstm_count = 0
    threshold_count = 0
    hybrid_count = 0

    aligned_entries = entries[WINDOW_SIZE:]

    with open(outfile, "w") as f:
        for i, score in enumerate(scores):
            ts = timestamps[i]
            entry = aligned_entries[i] if i < len(aligned_entries) else {}

            breaches = check_thresholds(entry)
            status = get_alert_status(score, breaches)

            if status is None:
                continue

            f.write(format_alert_line(ts, score, breaches, status) + "\n")

            if status == "HYBRID_ALERT":
                hybrid_count += 1
            elif status == "LSTM_ANOMALY":
                lstm_count += 1
            elif status == "THRESHOLD_ALERT":
                threshold_count += 1

        f.write(f"\nFinished anomaly scan.\n")
        f.write(f"LSTM only:      {lstm_count}\n")
        f.write(f"Threshold only: {threshold_count}\n")
        f.write(f"Hybrid alerts:  {hybrid_count}\n")
        f.write(f"Total:          {lstm_count + threshold_count + hybrid_count}\n")

    print(f"Anomalies written to {outfile}")
    print(f"LSTM only: {lstm_count} | Threshold only: {threshold_count} | Hybrid: {hybrid_count}")

def plot_scores(scores, outfile="anomaly_scores.png"):
    plt.figure(figsize=(14, 5))
    plt.plot(scores, label="Anomaly Score (1 min ahead)")
    plt.axhline(LSTM_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"LSTM Threshold ({LSTM_THRESHOLD:.4f})")
    plt.title("Predicted Anomaly Scores Over Time (60 Second Lookahead)")
    plt.xlabel("Window Index")
    plt.ylabel("Predicted Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Plot saved to {outfile}")

def run_live(path, outfile="anomalies.log"):
    print(f"Live mode started. Watching {path}...")
    print(f"Waiting for {WINDOW_SIZE} entries before scoring...")

    with open(outfile, "w") as f:
        f.write(f"Live anomaly detection started.\n")
        f.write(f"LSTM threshold: {LSTM_THRESHOLD:.6f}\n")
        f.write(f"CPU threshold:  {THRESHOLD_CPU}%\n")
        f.write(f"MEM threshold:  {THRESHOLD_MEMORY}%\n")
        f.write(f"NET_RX threshold: {THRESHOLD_NET_RX} B/s\n")
        f.write(f"NET_TX threshold: {THRESHOLD_NET_TX} B/s\n\n")

    last_scored_index = -1

    while True:
        entries = load_jsonl(path)
        total = len(entries)

        if total <= WINDOW_SIZE:
            print(f"\rWaiting for enough data... ({total}/{WINDOW_SIZE} entries)", end="", flush=True)
            time.sleep(5)
            continue

        if last_scored_index == -1:
            print(f"\nEnough data collected. Scoring started.")

        latest_index = total - WINDOW_SIZE - 1

        if latest_index <= last_scored_index:
            time.sleep(5)
            continue

        for i in range(last_scored_index + 1, latest_index + 1):
            window_entries = entries[i:i+WINDOW_SIZE]
            current_entry = entries[i+WINDOW_SIZE]
            ts = current_entry.get("timestamp_iso") or current_entry.get("timestamp")

            flattened = [flatten_entry(e) for e in window_entries]
            df = pd.DataFrame(flattened, columns=FEATURES)
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            scaled = scaler.transform(df)
            window = scaled

            score = score_window(window)
            breaches = check_thresholds(current_entry)
            status = get_alert_status(score, breaches)

            if status is not None:
                line = format_alert_line(ts, score, breaches, status)
                print(line)
                with open(outfile, "a") as f:
                    f.write(line + "\n")

        last_scored_index = latest_index
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection against a JSONL dataset")
    parser.add_argument("-i", "--input", required=True, help="Path to the JSONL dataset file")
    parser.add_argument("-b", "--batch", action="store_true", help="Use batch mode for faster scoring of large files")
    parser.add_argument("--live", action="store_true", help="Run in live mode, continuously watching the input file")
    args = parser.parse_args()

    if args.live and args.batch:
        print("Error: --live and --batch cannot be used together.")
        exit(1)

    entries = load_jsonl(args.input)
    print(f"Loaded {len(entries)} entries from {args.input}")

    if args.live:
        run_live(args.input)
    else:
        windows, timestamps = build_sequences(entries)
        print(f"Built {len(windows)} windows")
        print(f"Mode: {'batch' if args.batch else 'single'}")

        if args.batch:
            scores = compute_scores_batch(windows)
        else:
            scores = compute_scores_single(windows)

        print(f"LSTM Threshold: {LSTM_THRESHOLD:.6f}")
        print(f"Max score: {scores.max():.6f}")
        print(f"Mean score: {scores.mean():.6f}")

        write_anomalies(scores, timestamps, entries)
        plot_scores(scores)
