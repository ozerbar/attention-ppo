#!/bin/bash

# Hardcoded environment name ## SELECT YOUR ENVIRONMENT HERE ##
# ENV_NAME="HalfCheetah-v5"
ENV_NAME="Pusher-v5"

# Set desired observation repetition factor here
OBS_REPEAT=4

# ===================================================

# Create top-level runs directory if it doesn't exist
mkdir -p runs

BASE_ENV_DIR="runs/$ENV_NAME"

# Create base env directory
mkdir -p "$BASE_ENV_DIR"

# Find next free runN directory inside this env
i=1
while [ -d "$BASE_ENV_DIR/run$i" ]; do
  i=$((i+1))
done
RUN_BATCH_DIR="$BASE_ENV_DIR/run$i"

mkdir -p "$RUN_BATCH_DIR"
echo "Using run batch directory: $RUN_BATCH_DIR"

# Launch all seeds in parallel, passing ENV_NAME and RUN_BATCH_DIR to Python
for seed in 0 1 2
do
  ENV_NAME="$ENV_NAME" RUN_BATCH_DIR="$RUN_BATCH_DIR" python src/train.py --seed $seed --obs_repeat $OBS_REPEAT &
done

wait
