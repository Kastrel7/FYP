import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib

WINDOW_SIZE = 30
DATA_FILE = "dataset.jsonl"

def load_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass
    return entries

entries = load_jsonl(DATA_FILE)

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

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(ALL_FEATURES, "feature_order.pkl")

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

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")
plt.close()
