#!/bin/bash
echo "Starting stress test..."

CORES=$(nproc)
stress-ng --cpu $CORES --cpu-load 5 --cpu-method matrixprod --vm 1 --vm-bytes 512M --timeout 300 &
sleep 45
stress-ng --cpu $CORES --cpu-load 10 --cpu-method matrixprod --vm 1 --vm-bytes 1024M --timeout 255 &
sleep 45
stress-ng --cpu $CORES --cpu-load 15 --cpu-method matrixprod --vm 1 --vm-bytes 1536M --timeout 210 &
sleep 45
stress-ng --cpu $CORES --cpu-load 25 --cpu-method matrixprod --vm 1 --vm-bytes 2048M --timeout 165 &
sleep 45
stress-ng --cpu $CORES --cpu-load 30 --cpu-method matrixprod --vm 1 --vm-bytes 2560M --timeout 120

echo "Stress test complete."
