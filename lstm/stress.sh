#!/bin/bash
echo "Starting stress test..."

CORES=$(nproc)
stress-ng --cpu $CORES --cpu-load 5 --cpu-method matrixprod --timeout 300 &
sleep 30
stress-ng --cpu $CORES --cpu-load 10 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 270 &
sleep 20
stress-ng --cpu $CORES --cpu-load 15 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 250 &
sleep 20
stress-ng --cpu $CORES --cpu-load 25 --cpu-method matrixprod --vm 1 --vm-bytes 1024M --timeout 230 &
sleep 20
stress-ng --cpu $CORES --cpu-load 30 --cpu-method matrixprod --vm 1 --vm-bytes 2048M --timeout 210

echo "Stress test complete."
