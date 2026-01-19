#!/bin/bash

# Simple benchmarking script for distributed training scaling.

NUM_GPUS=(1 2 4)
MODEL_PATH="python/src/main.py" # Or distributed_train.py when integrated

echo "Starting Distributed Training Benchmark..."
echo "------------------------------------------"

for gpus in "${NUM_GPUS[@]}"; do
    echo "Running with $gpus GPUs..."
    
    # Note: torchrun handles the rank and world size environment variables
    START_TIME=$(date +%s%N)
    
    # Placeholder for actual command. Requires torchrun to be installed.
    # torchrun --nproc_per_node=$gpus $MODEL_PATH --benchmark --epochs 1
    
    END_TIME=$(date +%s%N)
    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 )) # ms
    
    echo "Time for $gpus GPUs: ${ELAPSED}ms"
done

echo "------------------------------------------"
echo "Benchmark Complete."
