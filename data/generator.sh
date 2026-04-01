#!/bin/bash

# Configuration
states=(10 15 25 50 75)  # Number of states
durations=(100 250 500 1000 5000)  # Maximum duration
timesteps=(1000 10000 100000 1000000)  # Number of timesteps
EMISSIONS=10  # Default or set your own
SEED=42

echo "Starting configuration generation and job submission..."

# Loop through all combinations
for s in "${states[@]}"; do
    for d in "${durations[@]}"; do
        for t in "${timesteps[@]}"; do

            # 1. Generate the JSON file using your Python script
            # We map: --n-states ($s), --max-duration ($d), --n-steps ($t)
            python3 data_generator.py \
                --n-states "$s" \
                --max-duration "$d" \
                --n-steps "$t" \
                --n-emissions "$EMISSIONS" \
                --seed "$SEED" \

            echo "-------------------------------------------"
        done
    done
done