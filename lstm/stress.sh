#!/bin/bash
echo "Starting stress test..."

stress-ng --cpu 1 --cpu-load 20 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 300 &
sleep 20
stress-ng --cpu 1 --cpu-load 20 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 280 &
sleep 20
stress-ng --cpu 1 --cpu-load 20 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 260 &
sleep 20
stress-ng --cpu 1 --cpu-load 20 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 240

echo "Stress test complete."
