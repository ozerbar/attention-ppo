#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
# Create top-level runs directory
mkdir -p runs

# SELECT ENVIRONMENT
ENV_NAME="AntBulletEnv-v0"  # <- set your environment name here
OBS_REPEAT=1  # <- set your observation repeat here
OBS_NOISE=0.0  # <- set your observation noise here

ANT_EXTRA_DIMS=0 # Adds extra random noise dimensions to the observation.
ANT_EXTRA_NOISE_STD=0.0

# WANDB logging flag
USE_WANDB=0  # set to 0 to disable wandb logging

# Use OBS_REPEAT, ANT_EXTRA_DIMS, and ANT_EXTRA_NOISE_STD in base env dir name
BASE_ENV_DIR="runs/${ENV_NAME}-x${OBS_REPEAT}-obs_noise_${OBS_NOISE}-extra_dims_${ANT_EXTRA_DIMS}-extra_std_${ANT_EXTRA_NOISE_STD}"
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
  ENV_NAME="$ENV_NAME" RUN_BATCH_DIR="$RUN_BATCH_DIR" USE_WANDB="$USE_WANDB" python src/train.py --seed $seed --obs_repeat $OBS_REPEAT --obs_noise $OBS_NOISE --extra_obs_dims $ANT_EXTRA_DIMS --extra_obs_noise_std $ANT_EXTRA_NOISE_STD &
done

wait
echo "All seeds finished."