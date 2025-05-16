#!/bin/bash

# Create runs directory if it doesn't exist
mkdir -p runs

# Find next free runN directory automatically
i=1
while [ -d "runs/run$i" ]; do
  i=$((i+1))
done
RUN_BATCH_DIR="runs/run$i"

mkdir -p "$RUN_BATCH_DIR"
echo "Using run batch directory: $RUN_BATCH_DIR"

# Launch all seeds in parallel, passing RUN_BATCH_DIR as environment variable
for seed in 0 1 2
do
  RUN_BATCH_DIR="$RUN_BATCH_DIR" python src/train_ppo.py --seed $seed &
done

wait
