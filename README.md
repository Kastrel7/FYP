# Hybrid Container Monitoring System for Self-Hosted Homelab Environments

A hybrid anomaly detection system that collects real Docker container and host metrics, trains an LSTM neural network on normal behaviour, and detects unusual activity using both machine learning and threshold-based rules simultaneously.

The system scores pre-collected JSONL datasets and produces timestamped anomaly logs and graphs as output.

---

## Requirements

- Docker and Docker Compose
- Python 3.12
- Native Linux environment (tested on Ubuntu 24.04)
- stress-ng installed on the host (for stress testing only)

Install Python dependencies from inside the `lstm` folder:

```bash
cd lstm
pip install -r requirements.txt
```

---

## Project Structure

```
docker/
    monitoring/
        prometheus/
            prometheus.yml
        docker-compose.yml
    workers/
        worker.py
        Dockerfile
        docker-compose.yml

lstm/
    metric_collector.py
    lstm_training.py
    anomaly_detection.py
    stress.sh
    requirements.txt
```

---

## How to Run

### Step 1: Start the Monitoring Stack

From the project root:

```bash
cd docker/monitoring
docker compose up -d
```

This starts Prometheus, cAdvisor, and Node Exporter. All three run on a shared internal Docker network and begin scraping metrics every 5 seconds.

---

### Step 2: Collect a Normal Usage Baseline

From the `lstm` folder, start the metric collector and let it run for at least 2 hours with the system under typical load.

```bash
cd lstm
python3 metric_collector.py -o normal.jsonl
```

For a more realistic baseline, run the worker containers during collection. From the `docker/workers` folder:

```bash
docker compose up --build -d
```
This starts two containers (`workload_a` and `workload_b`) that continuously vary their CPU and memory usage within defined ranges, giving the dataset a realistic spread of container behaviour.
A pre-collected dataset is available in the lstm folder as `normal.jsonl`.

---

### Step 3: Collect a Stressed Dataset

Start the metric collector into a new file, then trigger the stress test from a second terminal while it is running.

#### Terminal 1:

```bash
python3 metric_collector.py -o stressed.jsonl
```

#### Terminal 2 (after a few minutes of normal collection):
```bash
./stress.sh
```

Let the collector run for a few minutes after the stress test finishes to capture the recovery period, then stop it with Ctrl+C.
A pre-collected stressed dataset is available in the `lstm` folder as `stressed.jsonl`.

---

### Step 4: Train the LSTM Model

```bash
python3 lstm_training.py -i normal.jsonl
```

This will:

- Auto-detect all feature columns from the dataset
- Drop any features that are constant across the dataset
- Normalise values using MinMaxScaler calibrated to the true 0-100% range
- Build 30-step sliding windows (150 seconds of history per window)
- Train a two-layer stacked LSTM for 20 epochs
- Calculate the anomaly threshold from the training data using recursive 12-step prediction
- Save the following files to the current directory:

    - `lstm_model.keras` - the trained model
    - `scaler.pkl` - the fitted scaler
    - `feature_order.pkl` - the feature list used during training
    - `feature_std.pkl` - per-feature standard deviations used for score normalisation
    - `threshold_info.pkl` - the anomaly threshold, mean, and standard deviation
    - `training_loss.png` - a graph of training and validation loss across epochs

---

### Step 5: Run Anomaly Detection

```bash
python3 anomaly_detection.py -i stressed.jsonl
```

For faster processing use batch mode:

```bash
python3 anomaly_detection.py -i stressed.jsonl -b
```

This scores all windows in the dataset, writes detected anomalies to `anomalies.log`, and saves a graph of anomaly scores over time to `anomaly_scores.png`.

---

## Alert Types

The detection script produces three types of alert:

| Alert | Meaning |
|-------|---------|
| `LSTM_ANOMALY` | The LSTM score exceeded the threshold but no threshold rules fired |
| `THRESHOLD_ALERT` | A threshold rule fired but the LSTM score was below the threshold |
| `HYBRID_ALERT` | Both the LSTM score and at least one threshold rule fired simultaneously |

Example alert line:

```
2026-04-27T10:46:34.852511+00:00 | lstm_score=882.327765 | threshold=CPU(78.08%) | -> HYBRID_ALERT
```

---


## Output Files

| File | Description |
|------|-------------|
| `anomalies.log` | Timestamped log of all detected anomalies with LSTM scores and threshold breaches |
| `anomaly_scores.png` | Graph of anomaly scores over time with the threshold line marked |
| `training_loss.png` | Graph of training and validation loss across epochs |

---

## Threshold Rules (Default Values)

| Metric | Threshold |
|--------|-----------|
| CPU | > 75% |
| Memory | > 70% |
| Network RX | > 10,000 B/s |
| Network TX | > 50,000 B/s |

These can be adjusted at the top of `anomaly_detection.py`.

---