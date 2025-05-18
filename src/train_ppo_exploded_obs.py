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

# Parse the random seed and obs_repeat from command line
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
parser.add_argument("--obs_repeat", type=int, default=1, help="Observation repetition factor")
args = parser.parse_args()
SEED = args.seed
OBS_REPEAT = args.obs_repeat

# Get run batch directory from environment variable, fallback to default folder
RUN_BATCH_DIR = os.environ.get("RUN_BATCH_DIR")
if RUN_BATCH_DIR is None:
    # Fallback (not recommended if you want batch grouping)
    BASE_RUNS_DIR = "./runs"
    i = 1
    while os.path.exists(os.path.join(BASE_RUNS_DIR, f"run{i}")):
        i += 1
    RUN_BATCH_DIR = os.path.join(BASE_RUNS_DIR, f"run{i}")
    os.makedirs(RUN_BATCH_DIR, exist_ok=True)

# Create seed-specific folder inside batch folder
RUN_DIR = os.path.join(RUN_BATCH_DIR, f"PPO{SEED}")
os.makedirs(RUN_DIR, exist_ok=True)


ENV_NAME = os.environ.get("ENV_NAME")

# Initialize wandb
wandb.init(
    project=f"{ENV_NAME.lower()}-ppo",
    entity="adlr-01",
    name=f"ppo-seed-{SEED}",
    config={
        "policy_type": "MlpPolicy",
        "env_id": ENV_NAME,
        "total_timesteps": 5_000_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "clip_range": 0.2,
        "seed": SEED,
        "obs_repeat": OBS_REPEAT,
    }
)

config = wandb.config

# Save parameters.yaml
with open(os.path.join(RUN_DIR, "parameters.yaml"), "w") as f:
    yaml.dump(dict(config), f)

# Save command.txt
with open(os.path.join(RUN_DIR, "command.txt"), "w") as f:
    f.write("python " + " ".join(sys.argv) + "\n")


print(f"Creating environment: {config.env_id}")
try:
    # Try to use the argument if supported
    env = gym.make(config.env_id, exclude_current_positions_from_observation=False)  # optional: include root x-pos
except TypeError as e:
    if "unexpected keyword argument 'exclude_current_positions_from_observation'" in str(e):
        print(f"Warning: {config.env_id} does not support 'exclude_current_positions_from_observation'. Creating without it.")
        env = gym.make(config.env_id)
    else:
        raise  # re-raise other TypeErrors

# Create and seed the environment

env.reset(seed=SEED)
env = Monitor(env)

if OBS_REPEAT > 1:
    env = ObservationRepeater(env, repeat=OBS_REPEAT)

# Define the PPO model
model = PPO(
    config.policy_type,
    env,
    seed=SEED,
    learning_rate=config.learning_rate,
    n_steps=config.n_steps,
    batch_size=config.batch_size,
    gamma=config.gamma,
    clip_range=config.clip_range,
    verbose=1,
    tensorboard_log=os.path.join(RUN_DIR, "tensorboard"),
)

# Save every 100,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path=RUN_DIR,
    name_prefix="ppo_checkpoint"
)

# W&B callback
wandb_callback = WandbCallback(
    gradient_save_freq=10,
    model_save_path=RUN_DIR,
    verbose=2
)

# Combine both callbacks
callback = CallbackList([checkpoint_callback, wandb_callback])

# Train the model
model.learn(
    total_timesteps=config.total_timesteps,
    callback=callback
)

# Save final model
model.save(os.path.join(RUN_DIR, "ppo_halfcheetah_final"))

# Finish wandb run
wandb.finish()
