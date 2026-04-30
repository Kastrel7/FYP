import os
import warnings
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib

WINDOW_SIZE = 30
PREDICTION_HORIZON = 12

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass
    return entries

parser = argparse.ArgumentParser(description="Train LSTM model on a JSONL dataset")
parser.add_argument("-i", "--input", default="dataset.jsonl", help="Input JSONL file (default: dataset.jsonl)")
args = parser.parse_args()

print(f"Loading data from {args.input}...")
entries = load_jsonl(args.input)

host_features = set()
container_features = set()

for entry in entries:
    for hk in entry.get("host", {}).keys():
        host_features.add(f"host_{hk}")

    for cname, cdata in entry.get("containers", {}).items():
        metrics = cdata.get("metrics", {})
        for mk in metrics.keys():
            container_features.add(f"{cname}_{mk}")

host_features = sorted(list(host_features))
container_features = sorted(list(container_features))
ALL_FEATURES = host_features + container_features

print("Detected host features:", host_features)
print("Detected container features:", container_features)

def flatten(entry):
    row = {}

    for hk, val in entry.get("host", {}).items():
        row[f"host_{hk}"] = val

    for cname, cdata in entry.get("containers", {}).items():
        metrics = cdata.get("metrics", {})
        for mk, mval in metrics.items():
            row[f"{cname}_{mk}"] = mval

    for feat in ALL_FEATURES:
        if feat not in row:
            row[feat] = 0

    return row

rows = [flatten(e) for e in entries]
df = pd.DataFrame(rows, columns=ALL_FEATURES)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

constant = df.nunique()
constant_features = constant[constant <= 1].index.tolist()

if constant_features:
    print("Dropping constant features:", constant_features)
    df = df.drop(columns=constant_features)
    ALL_FEATURES = [f for f in ALL_FEATURES if f not in constant_features]

dummy_min = {f: 0.0 for f in ALL_FEATURES}
dummy_max = {f: 100.0 if 'percent' in f or 'usage' in f else 1000000.0 for f in ALL_FEATURES}

df_for_scaler = pd.concat([df, pd.DataFrame([dummy_min, dummy_max])], ignore_index=True)

scaler = MinMaxScaler()
scaler.fit(df_for_scaler)
scaled = scaler.transform(df)

feature_std = np.std(scaled, axis=0)
feature_std[feature_std < 1e-6] = 1e-6

joblib.dump(scaler, "scaler.pkl")
joblib.dump(ALL_FEATURES, "feature_order.pkl")
joblib.dump(feature_std, "feature_std.pkl")

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        seq = data[i:i+window]
        target = data[i+window]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, WINDOW_SIZE)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, len(ALL_FEATURES))),
    LSTM(32),
    Dense(len(ALL_FEATURES))
])

model.compile(optimizer="adam", loss="mse")
print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("lstm_model.keras")

def predict_ahead_batch(windows, horizon):
    """
    Recursively predict horizon steps ahead for all windows at once.
    Same logic as anomaly_detection.py so the threshold is calculated
    using identical scoring to what detection will use.
    """
    current_windows = np.array(windows).copy()

    for _ in range(horizon):
        next_steps = model.predict(current_windows, verbose=0)
        current_windows = np.roll(current_windows, -1, axis=1)
        current_windows[:, -1, :] = next_steps

    return next_steps

print("Calculating threshold from training data using recursive predictions...")
train_predictions = predict_ahead_batch(X_train, PREDICTION_HORIZON)
actual = X_train[:, -1, :]
train_scores = np.mean(np.abs(train_predictions - actual) / feature_std, axis=1)

threshold_mean = float(np.mean(train_scores))
threshold_std = float(np.std(train_scores))
threshold = threshold_mean + 3 * threshold_std

joblib.dump({
    "mean": threshold_mean,
    "std": threshold_std,
    "threshold": threshold
}, "threshold_info.pkl")

print(f"Threshold saved: {threshold:.6f} (mean={threshold_mean:.6f}, std={threshold_std:.6f})")

scale = 1e5
plt.figure(figsize=(10, 5))
plt.plot([v * scale for v in history.history["loss"]], label="train_loss")
plt.plot([v * scale for v in history.history["val_loss"]], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (×10⁻⁵)")
plt.title("Training Loss")
plt.savefig("training_loss.png")
plt.close()