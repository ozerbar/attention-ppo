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
from stable_baselines3.common.vec_env import VecEnvWrapper


class AddGaussianNoise(VecEnvWrapper):
    # Added by ozerbar
    def __init__(self, venv, sigma: float = 0.0, seed: int | None = None):
        super().__init__(venv)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Return *obs + N(0, sigma)* with the same shape as *obs*."""
        noise = self.rng.normal(0.0, self.sigma, size=obs.shape)
        return obs + noise
    # ---------------------------------------------------------------------
    # VecEnv‑required overrides
    # ---------------------------------------------------------------------
    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        return self._add_noise(obs)
    def step_wait(self): 
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._add_noise(obs), rewards, dones, infos



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

class AddExtraObsDims(VecEnvWrapper):
    """
    Append `extra_dims` i.i.d. N(0, std²) features to every observation.

    ▸ It is meant to be placed **after** VecNormalize so the new features are
      *not* normalised.

    ▸ The wrapper updates:
        • observations returned by `reset()` and `step_wait()`
        • `info["terminal_observation"]` / `info["final_observation"]`
          (needed by SB-3’s on-policy collectors)

    Example
    -------
    vec_norm = VecNormalize(raw_vec, norm_obs=True, norm_reward=False)
    env      = AddExtraObsDims(vec_norm, extra_dims=10, std=0.5)
    """

    def __init__(
        self,
        venv,
        extra_dims: int,
        std: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(venv)
        self.extra_dims = int(extra_dims)
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Extend the observation space with ±∞ bounds so SB-3 will not clip
        # ------------------------------------------------------------------
        low  = np.concatenate(
            [venv.observation_space.low,
             np.full(self.extra_dims, -np.inf)]
        )
        high = np.concatenate(
            [venv.observation_space.high,
             np.full(self.extra_dims,  np.inf)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=venv.observation_space.dtype)

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def _append(self, obs: np.ndarray) -> np.ndarray:
        """Return `obs` with extra noise features of matching batch shape."""
        if self.extra_dims == 0:
            return obs
        # Supports shapes (n_envs, obs_dim) and (obs_dim,)
        noise_shape = (*obs.shape[:-1], self.extra_dims)
        noise = self.rng.normal(0.0, self.std, size=noise_shape).astype(obs.dtype)
        return np.concatenate([obs, noise], axis=-1)

    def _patch_infos(self, infos):
        """Make sure terminal/final observations inside `infos` are patched."""
        for info in infos:
            for key in ("terminal_observation", "final_observation"):
                if key in info:
                    info[key] = self._append(info[key])
        return infos

    # ----------------------------------------------------------------------
    # VecEnv interface
    # ----------------------------------------------------------------------
    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)          # returns only obs
        return self._append(obs)

    def step_async(self, actions):
        # Delegate to the wrapped VecEnv
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs   = self._append(obs)
        infos = self._patch_infos(infos)
        return obs, rews, dones, infos
