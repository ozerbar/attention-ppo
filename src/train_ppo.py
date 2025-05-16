import os
import argparse
import sys
import yaml
import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

# Parse the random seed from command line
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
args = parser.parse_args()
SEED = args.seed

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

# Initialize wandb
wandb.init(
    project="halfcheetah-ppo",
    entity="adlr-01",
    name=f"ppo-seed-{SEED}",
    config={
        "policy_type": "MlpPolicy",
        "env_id": "HalfCheetah-v5",
        "total_timesteps": 2_000_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "clip_range": 0.2,
        "seed": SEED,
    }
)

config = wandb.config

# Save parameters.yaml
with open(os.path.join(RUN_DIR, "parameters.yaml"), "w") as f:
    yaml.dump(dict(config), f)

# Save command.txt
with open(os.path.join(RUN_DIR, "command.txt"), "w") as f:
    f.write("python " + " ".join(sys.argv) + "\n")

# Create and seed the environment
env = gym.make(config.env_id)
env.reset(seed=SEED)
env = Monitor(env)

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
    save_freq=100_000,
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
