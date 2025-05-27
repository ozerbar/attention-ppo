#!/bin/bash

ENV_NAME="AntBulletEnv-v0"
ALGO="ppo"
FOLDER="runs/AntBulletEnv-v0-zoo-repeat1-noise0/run1/seed0"

python -m rl_zoo3.enjoy \
  --env $ENV_NAME \
  --algo $ALGO \
  -f $FOLDER \
  --n-timesteps 10000 \
  --deterministic \
  --verbose 1 \
  --no-render # Rendering has problem with ssh
