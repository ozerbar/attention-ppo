import os
import sys
from pathlib import Path
import json
import importlib

import imageio
import gymnasium as gym
import numpy as np
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from observation_wrappers import (
    ObservationRepeater,
    AddGaussianNoise,
    AddExtraObsDims,
    AddExtraObsDimsRamp,
)

# ======== CONFIGURATION ========
MODEL_PATH = "runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_20-extra_std_1.0-frames_4-policy_MlpPolicy/run1/seed2/lunarlandercontinuous-v3-x1-seed2_final.zip"  # <-- UPDATE THIS!
ENV_NAME = "LunarLanderContinuous-v3"
SEED = 2
OUTPUT_VIDEO = f"LunarLanderContinuous-v3-x1-gaussian-extra_dims_20-extra_std_1.0-frames_4-MLPPolicy-SEED_{SEED}.mp4"
N_FRAMES = 500

# -------------------------------------------------------------------
# Load environment parameters that were saved during training
# -------------------------------------------------------------------
RUN_DIR = os.path.dirname(MODEL_PATH)
ENV_STACK_FILE = os.path.join(RUN_DIR, "env_stack.json")

# Default values match those in train.py
OBS_REPEAT = 1
OBS_NOISE = 0.0
EXTRA_OBS_TYPE = "linear"
EXTRA_OBS_DIMS = 0
EXTRA_OBS_NOISE_STD = 0.0
FRAME_STACK = 4
ENV_WRAPPER_CLS = None

if os.path.exists(ENV_STACK_FILE):
    with open(ENV_STACK_FILE, "r") as f:
        saved_params = json.load(f).get("parameters", {})

    OBS_REPEAT = saved_params.get("obs_repeat", OBS_REPEAT)
    OBS_NOISE = saved_params.get("obs_noise", OBS_NOISE)
    EXTRA_OBS_TYPE = saved_params.get("extra_obs_type", EXTRA_OBS_TYPE)
    EXTRA_OBS_DIMS = saved_params.get("extra_obs_dims", EXTRA_OBS_DIMS)
    EXTRA_OBS_NOISE_STD = saved_params.get("extra_obs_noise_std", EXTRA_OBS_NOISE_STD)
    FRAME_STACK = saved_params.get("frame_stack", FRAME_STACK)

    wrapper_path = saved_params.get("env_wrapper")
    if wrapper_path:
        mod, cls = wrapper_path.rsplit(".", 1)
        ENV_WRAPPER_CLS = getattr(importlib.import_module(mod), cls)

print("[INFO] Loaded render parameters:")
print(f"  OBS_REPEAT={OBS_REPEAT}, OBS_NOISE={OBS_NOISE}")
print(f"  EXTRA_OBS_DIMS={EXTRA_OBS_DIMS} ({EXTRA_OBS_TYPE}), FRAME_STACK={FRAME_STACK}")
print(f"  ENV_WRAPPER={ENV_WRAPPER_CLS}")

# -------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so unpickling can import `src.*`
# -------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# -------------------------------------------------------------------
# Helper to build the base (un-vectorised) environment
# -------------------------------------------------------------------

def _make_base_env():
    try:
        env = gym.make(
            ENV_NAME,
            render_mode="rgb_array",
            exclude_current_positions_from_observation=False,
        )
    except TypeError:  # some envs do not support the kwarg
        env = gym.make(ENV_NAME, render_mode="rgb_array")

    env.reset(seed=SEED)

    # Observation repetition (duplicates every obs entry `repeat` times)
    if OBS_REPEAT > 1:
        env = ObservationRepeater(env, repeat=OBS_REPEAT)

    # Apply any custom env wrapper that might have been used during training
    if ENV_WRAPPER_CLS is not None:
        env = ENV_WRAPPER_CLS(env)

    return env

# -------------------------------------------------------------------
# Re-create the exact wrapper chain used during training
# -------------------------------------------------------------------

# 1. DummyVecEnv for a single environment
vec_env = DummyVecEnv([_make_base_env])

# 2. Load VecNormalize statistics (if present)
vecnorm_path = os.path.join(RUN_DIR, "vecnormalize.pkl")
if os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, vec_env)
    env.training = False          # do not update running stats
    env.norm_reward = False
else:
    env = vec_env

# 3. Add Gaussian observation noise (post-normalisation)
if OBS_NOISE > 0.0:
    env = AddGaussianNoise(env, sigma=OBS_NOISE)

# 4. Append extra observation dimensions when they were used in training
if EXTRA_OBS_DIMS > 0:
    if EXTRA_OBS_TYPE.lower() == "gaussian":
        env = AddExtraObsDims(env, extra_dims=EXTRA_OBS_DIMS, std=EXTRA_OBS_NOISE_STD)
    else:
        env = AddExtraObsDimsRamp(
            env,
            extra_dims=EXTRA_OBS_DIMS,
            noise_type=EXTRA_OBS_TYPE,
            scale=0.1,
            period=50,
            amplitude=2.0,
        )

# 5. Temporal frame stacking
if FRAME_STACK > 1:
    env = VecFrameStack(env, n_stack=FRAME_STACK)

# === Load model ===
# Load model *with* environment to avoid n_env mismatch issues
model = PPO.load(MODEL_PATH, env=env)

# === Collect frames ===
frames = []
obs = env.reset()

for _ in trange(N_FRAMES):
    action, _states = model.predict(obs, deterministic=True)
    obs, _, dones, _ = env.step(action)

    # VecEnv.render() returns a list when n_envs > 1 (here n_envs == 1)
    frame = env.render()
    if isinstance(frame, (list, tuple)):
        frame = frame[0]

    if frame is not None:
        frames.append(frame)

    if dones[0]:
        obs = env.reset()

# === Save video ===
# -------------------------------------------------------------------
# Save the collected RGB frames as an mp4 video
# -------------------------------------------------------------------
imageio.mimsave(OUTPUT_VIDEO, frames, fps=30)
print(f"[INFO] Video saved to: {OUTPUT_VIDEO}")
# === Close environment ===