import multiprocessing
import time
import random
import psutil
import os

CPU_MIN = float(os.environ.get("CPU_MIN", 10))
CPU_MAX = float(os.environ.get("CPU_MAX", 30))
MEM_MIN = float(os.environ.get("MEM_MIN", 10))
MEM_MAX = float(os.environ.get("MEM_MAX", 30))
CHANGE_INTERVAL = float(os.environ.get("CHANGE_INTERVAL", 20))
CORES = int(float(os.environ.get("CORES", multiprocessing.cpu_count())))

def cpu_burn(target_percent, duration):
    end = time.time() + duration
    while time.time() < end:
        busy_end = time.time() + (target_percent / 100)
        while time.time() < busy_end:
            pass
        time.sleep(1 - (target_percent / 100))

def mem_burner(target_percent, duration):
    total_ram = psutil.virtual_memory().total
    target_bytes = int(total_ram * (target_percent / 100))
    end = time.time() + duration
    chunk = bytearray(target_bytes)
    while time.time() < end:
        time.sleep(0.1)
    del chunk

def stress(cpu_percent, mem_percent, duration=20):
    cpu_cores = int(CORES)

    if cpu_cores > multiprocessing.cpu_count():
        print(f"Warning: Requested {CORES} cores, but only {multiprocessing.cpu_count()} available. Using all available cores.")
        cpu_cores = multiprocessing.cpu_count()
    
    processes = []
    for _ in range(cpu_cores):
        p = multiprocessing.Process(target=cpu_burn, args=(cpu_percent, duration))
        p.start()
        processes.append(p)

    if mem_percent > 0:
        p = multiprocessing.Process(target=mem_burner, args=(mem_percent, duration))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("Done!")

def main():
    print(f"Starting worker: CPU {CPU_MIN}-{CPU_MAX}%, MEM {MEM_MIN}-{MEM_MAX}%, change every {CHANGE_INTERVAL}s")
    while True:
        target_cpu = random.uniform(CPU_MIN, CPU_MAX)
        target_mem = random.uniform(MEM_MIN, MEM_MAX)
        print(f"Target CPU: {target_cpu:.1f}%  MEM: {target_mem:.1f}%")
        stress(target_cpu, target_mem, CHANGE_INTERVAL)

if __name__ == "__main__":
    main()