import os
import argparse
import sys
import yaml
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401  (registers AntBullet‑v0)
import wandb
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.observation_wrappers import ObservationRepeater, ObservationNoiseAdder, ObservationNormalizer, AddGaussianNoise

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
parser.add_argument("--obs_repeat", type=int, default=1, help="Observation repetition factor")
parser.add_argument("--obs_noise", type=float, default=0.0, help="Std dev of Gaussian noise added to observations")
args = parser.parse_args()

SEED = args.seed
OBS_REPEAT = args.obs_repeat
OBS_NOISE = args.obs_noise
ENV_NAME = os.environ.get("ENV_NAME")
RUN_BATCH_DIR = os.environ.get("RUN_BATCH_DIR")
assert ENV_NAME is not None and RUN_BATCH_DIR is not None, "ENV_NAME and RUN_BATCH_DIR must be set!"

RUN_DIR = os.path.join(RUN_BATCH_DIR, f"seed{SEED}")
os.makedirs(RUN_DIR, exist_ok=True)
# Load hyperparameters from YAML: hyperparams/{ENV_NAME}.yml
hparams_path = os.path.join("hyperparams", f"{ENV_NAME}.yml")
with open(hparams_path, "r") as f:
    full_cfg = yaml.safe_load(f)

if ENV_NAME not in full_cfg:
    raise KeyError(f"{ENV_NAME} not found in {hparams_path}")

hp = full_cfg[ENV_NAME]
learning_rate = float(hp["learning_rate"])
n_steps       = int(hp["n_steps"])
batch_size    = int(hp["batch_size"])
gamma         = float(hp["gamma"])
clip_range    = float(hp["clip_range"])
ent_coef      = float(hp.get("ent_coef", 0.0))
vf_coef       = float(hp.get("vf_coef", 0.5))
gae_lambda    = float(hp.get("gae_lambda", 1.0))
max_grad_norm = float(hp.get("max_grad_norm", 0.5))
n_epochs      = int(hp.get("n_epochs", 10))
total_timesteps = int(hp.get("total_timesteps", 2_000_000))

policy_kwargs = {}
if "policy_kwargs" in hp:
    policy_kwargs = eval(
        hp["policy_kwargs"],
        {"__builtins__": None},
        {"nn": nn, "torch": torch, "dict": dict}
    )

# Construct a descriptive run name
run_name = (f"{ENV_NAME.lower()}-x{OBS_REPEAT}-seed{SEED}" + (f"-noise{OBS_NOISE}" if OBS_NOISE > 0 else ""))

# Initialize wandb
wandb.init(
    project=f"{ENV_NAME.lower()}-ppo",
    entity="adlr-01",
    name=run_name,
    config={
        **hp,
        "seed": SEED,
        "obs_repeat": OBS_REPEAT,
        "obs_noise": OBS_NOISE,
    },
)

config = wandb.config

# Save run metadata
with open(os.path.join(RUN_DIR, "parameters.yaml"), "w") as f:
    yaml.dump(dict(config), f)
with open(os.path.join(RUN_DIR, "command.txt"), "w") as f:
    f.write("python " + " ".join(sys.argv) + "\n")

# --- Environment creation using DummyVecEnv, VecNormalize, AddGaussianNoise ---
def make_env():
    def _thunk():
        try:
            try:
                env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)
            except TypeError:
                env = gym.make(ENV_NAME)
        except Exception as e:
            import gymnasium
            if isinstance(e, gymnasium.error.NameNotFound):
                print(f"ERROR: Environment '{ENV_NAME}' does not exist or is not installed. Please check your environment name and dependencies.")
                sys.exit(1)
            else:
                raise
        env.reset(seed=SEED)
        env = Monitor(env)
        if OBS_REPEAT > 1:
            env = ObservationRepeater(env, repeat=OBS_REPEAT)
        # Do not add noise here; will be handled by AddGaussianNoise after VecNormalize
        # if hp.get("normalize", False):
        #     env = ObservationNormalizer(env)
        # return env
    return _thunk

raw_vec = DummyVecEnv([make_env()])
vec_norm = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, training=True)
if OBS_NOISE > 0:
    env = AddGaussianNoise(vec_norm, sigma=OBS_NOISE)
else:
    env = vec_norm

# Initialize PPO model
model = PPO(
    hp.get("policy", "MlpPolicy"),
    env,
    seed=SEED,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    gamma=gamma,
    clip_range=clip_range,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    gae_lambda=gae_lambda,
    max_grad_norm=max_grad_norm,
    n_epochs=n_epochs,
    policy_kwargs=policy_kwargs,
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
    verbose=2,
    log="all",
)

callback = CallbackList([checkpoint_callback, wandb_callback])

# Train the agent
model.learn(
    total_timesteps=total_timesteps,
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
