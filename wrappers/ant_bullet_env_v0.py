import os
import numpy as np
import gymnasium as gym

class ObservationRepeater(gym.ObservationWrapper):
    def __init__(self, env, repeat=1):
        super().__init__(env)
        self.repeat = repeat
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, repeat),
            high=np.tile(env.observation_space.high, repeat),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return np.tile(obs, self.repeat)


class ObservationNoiseAdder(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0.0, self.noise_std, size=obs.shape)
        return obs + noise


from sb3_contrib.common.wrappers import TimeFeatureWrapper

def wrap_ant(env):
    # print("Custom wrap_ant() called")

    # Always apply TimeFeatureWrapper first
    env = TimeFeatureWrapper(env)

    obs_repeat = int(os.environ.get("ANT_OBS_REPEAT", 1))
    obs_noise = float(os.environ.get("ANT_OBS_NOISE", 0.0))

    if obs_repeat > 1:
        env = ObservationRepeater(env, repeat=obs_repeat)
        # print(f"Applied ObservationRepeater with repeat={obs_repeat}")
    if obs_noise > 0.0:
        env = ObservationNoiseAdder(env, noise_std=obs_noise)
        # print(f"Applied ObservationNoiseAdder with noise_std={obs_noise}")
    return env

