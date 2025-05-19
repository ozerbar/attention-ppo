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

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
parser.add_argument("--obs_repeat", type=int, default=1, help="Observation repetition factor")
args = parser.parse_args()

SEED = args.seed
OBS_REPEAT = args.obs_repeat
ENV_NAME = os.environ.get("ENV_NAME")
RUN_BATCH_DIR = os.environ.get("RUN_BATCH_DIR")
assert ENV_NAME is not None and RUN_BATCH_DIR is not None, "ENV_NAME and RUN_BATCH_DIR must be set!"

RUN_DIR = os.path.join(RUN_BATCH_DIR, f"seed{SEED}")
os.makedirs(RUN_DIR, exist_ok=True)

# Construct a descriptive run name
run_name = f"{ENV_NAME.lower()}-x{OBS_REPEAT}-ppo-seed{SEED}"

# Initialize wandb
wandb.init(
    project=f"{ENV_NAME.lower()}-ppo",
    entity="adlr-01",
    name=run_name,
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

# Save run metadata
with open(os.path.join(RUN_DIR, "parameters.yaml"), "w") as f:
    yaml.dump(dict(config), f)
with open(os.path.join(RUN_DIR, "command.txt"), "w") as f:
    f.write("python " + " ".join(sys.argv) + "\n")

print(f"Creating environment: {ENV_NAME}")
try:
    env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)
except TypeError:
    env = gym.make(ENV_NAME)

env.reset(seed=SEED)
env = Monitor(env)

if OBS_REPEAT > 1:
    env = ObservationRepeater(env, repeat=OBS_REPEAT)

# Initialize PPO model
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

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path=RUN_DIR,
    name_prefix="checkpoint"
)

wandb_callback = WandbCallback(
    gradient_save_freq=10,
    model_save_path=RUN_DIR,
    verbose=2
)

callback = CallbackList([checkpoint_callback, wandb_callback])

# Train the agent
model.learn(
    total_timesteps=config.total_timesteps,
    callback=callback
)

# Save and upload final model
final_model_path = os.path.join(RUN_DIR, f"{run_name}_final.zip")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

artifact = wandb.Artifact(name=run_name, type="model")
artifact.add_file(final_model_path)
wandb.log_artifact(artifact)

wandb.finish()
