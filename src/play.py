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
from observation_wrappers import (
    ObservationRepeater,
    AddGaussianNoise,
    AddExtraObsDims,
)


# -------------------------------
# Define run directory
# -------------------------------
RELATIVE_DIR = "runs/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_0-extra_std_0.0-frames_10"

RUN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", RELATIVE_DIR, "run1/seed0"))
EPISODES = 5



# -------------------------------
# Automatically find model file (.zip)
# -------------------------------
model_candidates = [f for f in os.listdir(RUN_DIR) if f.endswith(".zip")]
model_candidates = [os.path.join(RUN_DIR, f) for f in model_candidates]
if not model_candidates:
    raise FileNotFoundError(f"No model (.zip) found in {RUN_DIR}")
MODEL_PATH = model_candidates[0]

# Fixed paths for saved vecnormalize and stack
VECNORM_PATH = os.path.join(RUN_DIR, "vecnormalize.pkl")
STACK_PATH = os.path.join(RUN_DIR, "env_stack.json")

# -------------------------------
# Load env config
# -------------------------------
with open(STACK_PATH, "r") as f:
    meta = json.load(f)

params = meta["parameters"]
wrapper_stack = meta["wrapper_stack"]

ENV_NAME = wrapper_stack[-1]

OBS_REPEAT = params.get("obs_repeat", 1)
OBS_NOISE = params.get("obs_noise", 0.0)
EXTRA_OBS_DIMS = params.get("extra_obs_dims", 0)
EXTRA_OBS_NOISE_STD = params.get("extra_obs_noise_std", 0.0)
FRAME_STACK = params.get("frame_stack", 1)

env_wrapper_path = params.get("env_wrapper", None)
env_wrapper_cls = None
if env_wrapper_path:
    modname, classname = env_wrapper_path.rsplit(".", 1)
    env_wrapper_cls = getattr(__import__(modname, fromlist=[classname]), classname)

# -------------------------------
# Environment reconstruction
# -------------------------------
def make_env():
    def _thunk():
        try:
            if "exclude_current_positions_from_observation" in params:
                env = gym.make(ENV_NAME, exclude_current_positions_from_observation=params["exclude_current_positions_from_observation"])
            else:
                env = gym.make(ENV_NAME)  # Inference version: don't pass this argument
        except Exception as e:
            if isinstance(e, gym.error.NameNotFound):
                print(f"ERROR: Environment '{ENV_NAME}' does not exist or is not installed. Please check your environment name and dependencies.")
                sys.exit(1)
            else:
                raise
        env.reset()
        env = Monitor(env)
        if OBS_REPEAT > 1:
            env = ObservationRepeater(env, repeat=OBS_REPEAT)
        # Apply env_wrapper if specified
        if env_wrapper_cls is not None:
            env = env_wrapper_cls(env)  # TimeFeatureWrapper in AntBullet-v0
        return env
    return _thunk


vec_env = DummyVecEnv([make_env()])
if FRAME_STACK > 1:
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)

vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
vec_env.training = False
vec_env.norm_reward = False

if OBS_NOISE > 0:
    vec_env = AddGaussianNoise(vec_env, sigma=OBS_NOISE)

if EXTRA_OBS_DIMS > 0:
    env = AddExtraObsDims(vec_env, extra_dims=EXTRA_OBS_DIMS, std=EXTRA_OBS_NOISE_STD)
else:
    env = vec_env

# -------------------------------
# Load model
# -------------------------------
model = PPO.load(MODEL_PATH)

# -------------------------------
# Run inference
# -------------------------------
for ep in range(EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step += 1

    print(f"Episode {ep+1}: total_reward = {total_reward:.2f}, steps = {step}")
