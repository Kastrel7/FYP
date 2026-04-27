# Hybrid Container Monitoring System for Self-Hosted Homelab Environments

A hybrid anomaly detection system that collects real Docker container and host metrics, trains an LSTM neural network on normal behaviour, and detects unusual activity using both machine learning and threshold-based rules simultaneously.

The system supports live detection mode, where alerts are printed to the terminal in real time as new data arrives, and file mode for scoring pre-collected datasets.

---

## Requirements

- Docker and Docker Compose
- Python 3.12+
- WSL (Ubuntu) or a native Linux environment
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

From the `lstm` folder, start the metric collector and let it run for at least 2 hours under typical system activity. The more varied the activity during collection, the better the model will generalise.

```bash
cd lstm
python3 metric_collector.py -o normal.jsonl
```

Each sample is written as a single line to the output file every 5 seconds. The script automatically discovers all running non-monitoring containers and collects CPU, memory, and network metrics for each one alongside host-level metrics.

---

### Step 3: Train the LSTM Model

Once you have collected enough data, train the model:

```bash
python3 lstm_training.py -i normal.jsonl
```

This will:

- Auto-detect all feature columns from the dataset
- Drop any features that are constant across the dataset
- Normalise values using MinMaxScaler
- Build 30-step sliding windows (150 seconds of history per window)
- Train a two-layer stacked LSTM for 20 epochs
- Calculate the anomaly threshold from the training data using recursive 12-step prediction
- Save the following files to the current directory:
  - `lstm_model.keras` - the trained model
  - `scaler.pkl` - the fitted scaler
  - `feature_order.pkl` - the feature list used during training
  - `threshold_info.pkl` - the anomaly threshold, mean, and standard deviation
  - `training_loss.png` - a graph of training and validation loss across epochs

---

### Step 4: Run Live Detection

Open two terminals inside the `lstm` folder.

**Terminal 1** - start collecting live metrics:

```bash
python3 metric_collector.py -o live.jsonl
```

**Terminal 2** - start live hybrid detection:

```bash
python3 anomaly_detection.py -i live.jsonl --live
```

The detection script will wait until 30 entries have been collected before scoring begins, showing progress while it waits. Once scoring starts, any detected anomalies are printed to the terminal immediately and written to `anomalies.log`.

---

### Step 5: Run the Stress Test (Optional)

From a third terminal inside the `lstm` folder:

```bash
./stress.sh
```

The stress test uses stress-ng to ramp CPU and memory load up gradually across five stages over approximately 5 minutes. It uses all available CPU cores detected via `nproc`. You should start seeing LSTM anomaly alerts in Terminal 2 shortly after the stress test begins, with hybrid alerts appearing as CPU crosses the 75% threshold.

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
2026-04-27T10:46:34.852511+00:00 | lstm_score=0.581538 | threshold=CPU(76.37%) | -> HYBRID_ALERT
```

---

## File Mode (Scoring Pre-Collected Data)

To score a pre-collected JSONL file without live detection:

```bash
python3 anomaly_detection.py -i stressed.jsonl
```

For faster processing of large files, use batch mode:

```bash
python3 anomaly_detection.py -i stressed.jsonl -b
```

File mode produces `anomalies.log` and `anomaly_scores.png` in the current directory.

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