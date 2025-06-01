import os
import numpy as np
import gymnasium as gym
from src.observation_wrappers import ObservationRepeater, ObservationNoiseAdder, ObservationNormalizer


from sb3_contrib.common.wrappers import TimeFeatureWrapper

def wrap_ant(env):
    # print("Custom wrap_ant() called")

    # Always apply TimeFeatureWrapper first
    env = TimeFeatureWrapper(env)

    obs_repeat = int(os.environ.get("OBS_REPEAT", 1))
    obs_noise = float(os.environ.get("OBS_NOISE", 0.0))
    extra_dims = int(os.environ.get("ANT_EXTRA_DIMS", 0))
    extra_noise_std = float(os.environ.get("ANT_EXTRA_NOISE_STD", 0.0))

    if obs_repeat > 1:
        env = ObservationRepeater(env, repeat=obs_repeat)
        # print(f"Applied ObservationRepeater with repeat={obs_repeat}")
    if obs_noise > 0.0:
        env = ObservationNoiseAdder(env, noise_std=obs_noise)
        # print(f"Applied ObservationNoiseAdder with noise_std={obs_noise}")
    if extra_dims > 0:
        env = ObservationDimExpander(env, extra_dims=extra_dims, noise_std=extra_noise_std)

    return env

