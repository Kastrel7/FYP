# Hybrid Monitoring System for Self-Hosted Homelab Environments

A lightweight system that collects Docker container metrics, trains an LSTM model, and detects anomalous behaviour over time.  
This project includes:

- A monitoring stack (Docker Compose) for collecting container and host metrics
- Scripts for metric collection, model training, and anomaly detection

## Requirements

Before running the project, install the required Python dependencies:

```sh
pip install -r requirements.txt
```



- Docker and Docker Compose installed  
- Python 3.12+  
- At least one running container (excluding the monitoring stack) to generate behaviour  
- Recommended: run containers under different loads for better training data

## 1. Start the Monitoring Stack

From the project root:

```sh
cd docker/monitoring
docker compose up -d
```

This launches the monitoring services required by the metric collector.

If your environment differs (different ports, interfaces, or paths), configuration files inside the `monitoring` folder can be edited to match your setup.

## 2. Run Test Containers

Start any Docker containers you want to monitor.  
More varied workloads = better training data.

## 3. Collect Metrics

Inside the `lstm` folder:

```sh
cd lstm
python3 metric_collector.py
```

Let the script run for a period of time.  
Try collecting data under different loads.  
Each run produces a `.jsonl` file in the directory.

## 4. Train the LSTM Model

Once you have one or more `.jsonl` metric files, run:

```sh
python3 lstm_training.py
```

The script will:

- Detect all `.jsonl` files in the folder  
- Prepare the dataset  
- Train an LSTM model  
- Save the model and scaler

## 5. Detect Anomalies

Select a `.jsonl` file you previously recorded (or collect a new one), then run:

```sh
python3 anomaly_detection.py dataset.jsonl
```

This will:

- Load the trained LSTM model  
- Compute anomaly scores
- Generate a log file containing timestamps and anomaly scores
- Produce a graph visualising anomaly scores over time

Outputs will be saved in the same directory.

## Output Files

- **anomalies.log** – timestamps + anomaly scores  
- **anomaly_scores.png** – graph of anomaly score progression  

## Project Structure

```
docker/
    monitoring/
        loki/
            loki.yml
        prometheus/
            prometheus.yml
        promtail/
            promtail.yml
        docker-compose.yml

lstm/
    anomaly_detection.py
    lstm_training.py
    metric_collector.py
    requirments.txt
```

## Notes

- Training should be repeated if your workload changes significantly  
- The more varied your training data, the more accurate the anomaly detection becomes  
