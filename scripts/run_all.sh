#!/bin/sh
export PYTHONPATH="$PYTHONPATH:$(pwd)"
# Create top-level runs directory
mkdir -p runs

# SELECT ENVIRONMENT
ENV_NAME="AntBulletEnv-v0"  # <- set your environment name here
OBS_REPEAT=1                # <- set your observation repeat here
OBS_NOISE=0.0               # <- set your observation noise here
EXTRA_OBS_DIMS=0            # Adds extra random noise dimensions to the observation.
EXTRA_OBS_NOISE_STD=0.0
FRAME_STACK=4               # Number of frames to stack in the observation
FRAME_STACK=4               # Number of frames to stack in the observation

# Attention flags
ATTN_ACT=true
ATTN_VAL=true
ATTN_COMMON=false  # Use common attention for both actor and critic

ATTN_DIRECT_OVERRIDE=false  # Use direct override for attention blocks


# WANDB logging flag
USE_WANDB=0  # set to 0 to disable wandb logging

# --- Determine policy, config file, and run tags based on attention flags ---
POLICY="MlpPolicy"
CONF_FILE="hyperparams/${ENV_NAME}.yml"
ATTN_TAG=""
if [ "$ATTN_ACT" = "true" ] || [ "$ATTN_VAL" = "true" ] || [ "$ATTN_COMMON" = "true" ]; then
    if [ "$ATTN_DIRECT_OVERRIDE" = "true" ]; then
        POLICY="AttentionDirectOverridePolicy"
        CONF_FILE="hyperparams/${ENV_NAME}-attn-direct-override.yml"
        ATTN_TAG="-attn_direct_override${ATTN_DIRECT_OVERRIDE}-attn_val${ATTN_VAL}-attn_common${ATTN_COMMON}"
    else
        POLICY="MediumAttentionPolicy"
        CONF_FILE="hyperparams/${ENV_NAME}-attn-direct-override.yml"
        ATTN_TAG="-attn_act${ATTN_ACT}-attn_val${ATTN_VAL}-attn_common${ATTN_COMMON}"
    fi
fi

# --- Construct arguments for train.py ---
set -- --obs_repeat "$OBS_REPEAT" \
    --obs_noise "$OBS_NOISE" \
    --frame_stack "$FRAME_STACK" \
    --extra_obs_dims "$EXTRA_OBS_DIMS" \
    --extra_obs_noise_std "$EXTRA_OBS_NOISE_STD" \
    --policy "$POLICY" \
    --conf-file "$CONF_FILE"

if [ "$ATTN_ACT" = "true" ]; then
    set -- "$@" --attn_act
fi
if [ "$ATTN_VAL" = "true" ]; then
    set -- "$@" --attn_val
fi
if [ "$ATTN_COMMON" = "true" ]; then
    set -- "$@" --attn_common
fi

# --- Set up run directory ---
BASE_ENV_DIR="runs/${ENV_NAME}/${ENV_NAME}-x${OBS_REPEAT}-obs_noise_${OBS_NOISE}-extra_dims_${EXTRA_OBS_DIMS}-extra_std_${EXTRA_OBS_NOISE_STD}-frames_${FRAME_STACK}${ATTN_TAG}-policy_${POLICY}"
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
for seed in 0
do
  echo "Launching seed $seed"
  # Pass environment variables and arguments to the training script
  ENV_NAME="$ENV_NAME" RUN_BATCH_DIR="$RUN_BATCH_DIR" USE_WANDB="$USE_WANDB" \
  python src/train.py --seed "$seed" "$@" &
done

wait
echo "All seeds finished."