import time
import json
from datetime import datetime, timezone

import requests
import docker

PROM_URL = "http://localhost:9090/api/v1/query"
LOKI_URL = "http://localhost:3100/loki/api/v1/query_range"
OUTPUT_FILE = "dataset.jsonl"
SAMPLE_INTERVAL_SECONDS = 5

MONITORING_STACK = "monitoring"

def prom_query(query):
    """Run a PromQL query and return the scalar value, or None."""
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
        '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[30s])) * 100)'
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
    """
    cAdvisor exposes containers as:
    /system.slice/docker-<cid>.scope

    So we must match:
    id=~".*/system.slice/docker-<cid>.*\\.scope"
    """
    
    container_system_slice = f'/system.slice/docker-{container_id}.scope'

    cpu = prom_query(
        f'rate(container_cpu_usage_seconds_total{{id="{container_system_slice}"}}[30s]) * 100'
    )

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

def fetch_container_logs_from_loki(container_name, seconds=5):
    return []

def get_docker_client():
    return docker.DockerClient(base_url="unix:///var/run/docker.sock")


def discover_active_containers(client):
    """Return all non-monitoring containers."""
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
        logs = fetch_container_logs_from_loki(name, seconds=SAMPLE_INTERVAL_SECONDS)

        container_data[name] = {
            "metrics": metrics,
            "logs": logs,
        }

    return {
        "timestamp": ts,
        "timestamp_iso": iso,
        "host": host_metrics,
        "containers": container_data,
    }
    
def r(x):
    return round(x, 2) if x is not None else None

def main():
    client = get_docker_client()
    print(f"Collecting metrics every {SAMPLE_INTERVAL_SECONDS}s…")

    while True:
        sample = collect_sample(client)

        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(sample) + "\n")

        time.sleep(SAMPLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
