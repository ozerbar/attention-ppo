import os
import argparse
import sys
import yaml
import numpy as np
import importlib
import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401  (registers AntBullet‑v0)
import wandb
import torch
import torch.nn as nn
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack


ENV_NAME = "AntBulletEnv-v0"  
SEED = int(os.environ.get("SEED", 0))
OBS_REPEAT = int(os.environ.get("OBS_REPEAT", 1))
OBS_NOISE = float(os.environ.get("OBS_NOISE", 0.0))
EXTRA_OBS_DIMS = int(os.environ.get("EXTRA_OBS_DIMS", 0))
EXTRA_OBS_NOISE_STD = float(os.environ.get("EXTRA_OBS_NOISE_STD", 0.0))
FRAME_STACK = int(os.environ.get("FRAME_STACK", 1))

assert ENV_NAME is not None, "ENV_NAME must be set!"

def make_env():
    try:
        try:
            env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)
        except TypeError:
            env = gym.make(ENV_NAME)
    except Exception as e:
        if isinstance(e, gym.error.NameNotFound):
            print(f"ERROR: Environment '{ENV_NAME}' does not exist or is not installed. Please check your environment name and dependencies.")
            sys.exit(1)
        else:
            raise
    env.reset()
    env = Monitor(env)
    return env

env = make_env()

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
# print("env.unwrapped attributes:", dir(env.unwrapped))

