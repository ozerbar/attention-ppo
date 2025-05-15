import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

wandb.init(
    project="halfcheetah-ppo",
    entity="adlr-01",
    config={
        "policy_type": "MlpPolicy",
        "env_id": "HalfCheetah-v4",
        "total_timesteps": 100_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "clip_range": 0.2,
    }
)

config = wandb.config

env = gym.make(config.env_id)
env = Monitor(env)

model = PPO(
    config.policy_type,
    env,
    learning_rate=config.learning_rate,
    n_steps=config.n_steps,
    batch_size=config.batch_size,
    gamma=config.gamma,
    clip_range=config.clip_range,
    verbose=1
)

model.learn(
    total_timesteps=config.total_timesteps,
    callback=WandbCallback(gradient_save_freq=10, model_save_path="./models", verbose=2)
)
model.save("ppo_halfcheetah")

wandb.finish()
