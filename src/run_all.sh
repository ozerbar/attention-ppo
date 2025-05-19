#!/bin/bash

# Create top-level runs directory
mkdir -p runs

# SELECT ENVIRONMENT
ENV_NAME="Pusher-v5"
OBS_REPEAT=2  # <- set your observation repeat here

# Use OBS_REPEAT in base env dir name
BASE_ENV_DIR="runs/${ENV_NAME}-x${OBS_REPEAT}"
mkdir -p "$BASE_ENV_DIR"

# Find next available runN folder
i=1
while [ -d "$BASE_ENV_DIR/run$i" ]; do
  i=$((i+1))
done
RUN_BATCH_DIR="$BASE_ENV_DIR/run$i"
mkdir -p "$RUN_BATCH_DIR"

echo "Using run batch directory: $RUN_BATCH_DIR"

# Launch seeds
for seed in 0 1 2
do
  echo "Launching seed $seed"
  ENV_NAME="$ENV_NAME" RUN_BATCH_DIR="$RUN_BATCH_DIR" python src/train.py --seed $seed --obs_repeat $OBS_REPEAT &
done

wait
echo "All seeds finished."