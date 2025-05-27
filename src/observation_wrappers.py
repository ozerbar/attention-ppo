import os
import argparse
import sys
import yaml
import numpy as np
import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

class ObservationRepeater(gym.ObservationWrapper):
    def __init__(self, env, repeat=2):
        super().__init__(env)
        self.repeat = repeat
        low = np.tile(env.observation_space.low, self.repeat)
        high = np.tile(env.observation_space.high, self.repeat)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, obs):
        return np.tile(obs, self.repeat)

class ObservationNoiseAdder(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space

    def observation(self, obs):
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=obs.shape)
        return obs + noise
    
class ObservationNormalizer(gym.ObservationWrapper):
    """
    Min-max normalize each entry to [-1,1] using finite bounds only.
    Infinite or zero-range dimensions map to zero mean and unit scale.
    """
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        # replace infinities with zeros for mid/scale computation
        finite_low = np.where(np.isfinite(low), low, 0.0)
        finite_high = np.where(np.isfinite(high), high, 0.0)
        mid = (finite_high + finite_low) / 2.0
        scale = (finite_high - finite_low) / 2.0
        # avoid zero scale
        scale = np.where(scale <= 0.0, 1.0, scale)
        self.mid = mid
        self.scale = scale
        # output space unchanged
        self.observation_space = env.observation_space

    def observation(self, obs):
        normed = (obs - self.mid) / (self.scale + 1e-8)
        # replace any NaNs/infs from obs outside finite range
        return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=-1.0)
    
class ObservationDimExpander(gym.ObservationWrapper):
    """
    Adds extra random noise dimensions to the observation.
    """
    def __init__(self, env, extra_dims=0, noise_std=1.0):
        super().__init__(env)
        self.extra_dims = extra_dims
        self.noise_std = noise_std

        low = np.concatenate([env.observation_space.low, -np.ones(extra_dims)])
        high = np.concatenate([env.observation_space.high, np.ones(extra_dims)])

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, obs):
        extra_noise = np.random.normal(0.0, self.noise_std, size=self.extra_dims)
        return np.concatenate([obs, extra_noise])
