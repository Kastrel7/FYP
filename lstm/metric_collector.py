import multiprocessing
import time
import json
import argparse
from datetime import datetime, timezone

import requests
import docker

PROM_URL = "http://localhost:9090/api/v1/query"
SAMPLE_INTERVAL_SECONDS = 5

MONITORING_STACK = "monitoring"

def prom_query(query):
    try:
        r = requests.get(PROM_URL, params={"query": query}, timeout=3)
        r.raise_for_status()
        data = r.json()

        if data["status"] != "success":
            return None

        results = data["data"]["result"]
        if not results:
            return None

        return float(results[0]["value"][1])
    except Exception:
        return None

def get_host_metrics_prometheus():
    cpu = prom_query(
        'clamp_min(100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[30s])) * 100), 0)'
    )
    memory_usage = prom_query(
        '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)'
    )
    memory_limit = prom_query('node_memory_MemTotal_bytes')
    net_rx = prom_query('sum(rate(node_network_receive_bytes_total[30s]))')
    net_tx = prom_query('sum(rate(node_network_transmit_bytes_total[30s]))')
    memory_percent = None
    if memory_usage is not None and memory_limit and memory_limit > 0:
        memory_percent = (memory_usage / memory_limit) * 100
    return {
        "cpu_percent": r(cpu),
        "memory_usage": r(memory_percent),
        "net_rx": r(net_rx),
        "net_tx": r(net_tx),
    }

def get_container_metrics(container_id):
    num_cores = multiprocessing.cpu_count()
    container_system_slice = f'/system.slice/docker-{container_id}.scope'
    cpu = prom_query(
        f'rate(container_cpu_usage_seconds_total{{id="{container_system_slice}"}}[30s]) * 100'
    )
    if cpu is not None:
        cpu = cpu / num_cores

    memory_usage = prom_query(
        f'container_memory_usage_bytes{{id="{container_system_slice}"}}'
    )
    
    memory_limit = prom_query('node_memory_MemTotal_bytes')
    
    net_rx = prom_query(
        f'rate(container_network_receive_bytes_total{{id="{container_system_slice}"}}[30s])'
    )

    net_tx = prom_query(
        f'rate(container_network_transmit_bytes_total{{id="{container_system_slice}"}}[30s])'
    )

    memory_percent = None
    if memory_usage is not None and memory_limit and memory_limit > 0:
        memory_percent = (memory_usage / memory_limit) * 100

    return {
        "cpu_percent": r(cpu),
        "memory_usage": r(memory_percent),
        "net_rx": r(net_rx),
        "net_tx": r(net_tx),
    }

def get_docker_client():
    return docker.DockerClient(base_url="unix:///var/run/docker.sock")

def discover_active_containers(client):
    result = {}
    for c in client.containers.list():
        project = c.labels.get("com.docker.compose.project", "").lower()
        if project == MONITORING_STACK:
            continue
        result[c.name] = c
    return result

def collect_sample(client):
    ts = time.time()
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    host_metrics = get_host_metrics_prometheus()
    containers = discover_active_containers(client)

    container_data = {}

    for name, obj in containers.items():
        metrics = get_container_metrics(obj.id)
        container_data[name] = {"metrics": metrics}

    return {
        "timestamp": ts,
        "timestamp_iso": iso,
        "host": host_metrics,
        "containers": container_data,
    }

def r(x):
    return round(x, 2) if x is not None else None

def main():
    parser = argparse.ArgumentParser(description="Collect container and host metrics into a JSONL dataset")
    parser.add_argument("-o", "--output", default="dataset.jsonl", help="Output JSONL file (default: dataset.jsonl)")
    args = parser.parse_args()

    client = get_docker_client()
    print(f"Collecting metrics every {SAMPLE_INTERVAL_SECONDS}s -> {args.output}")

    while True:
        sample = collect_sample(client)

        with open(args.output, "a") as f:
            f.write(json.dumps(sample) + "\n")

        time.sleep(SAMPLE_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
