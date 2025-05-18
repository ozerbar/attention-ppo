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

# Set desired observation repetition factor here
OBS_REPEAT=2

# Launch all seeds in parallel, passing RUN_BATCH_DIR and obs_repeat as env and argument
for seed in 0 1 2
do
  RUN_BATCH_DIR="$RUN_BATCH_DIR" python src/train_ppo_exploded_obs.py --seed $seed --obs_repeat $OBS_REPEAT &
done

wait
