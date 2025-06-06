import os
import argparse
import sys
import yaml
import numpy as np
import importlib
import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401  (registers AntBullet‑v0)
import wandb
import torch
import torch.nn as nn
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from src.observation_wrappers import (
    ObservationRepeater,
    AddGaussianNoise,
    AddExtraObsDims,
)

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
parser.add_argument("--obs_repeat", type=int, default=1, help="Observation repetition factor")
parser.add_argument("--obs_noise", type=float, default=0.0, help="Std dev of Gaussian noise added to observations")
parser.add_argument("--extra_obs_dims", type=int, default=0, help="Number of extra random noise dims to add to observation")
parser.add_argument("--extra_obs_noise_std", type=float, default=0.0, help="Stddev of extra random noise dims")
parser.add_argument("--frame_stack", type=int, default=1, help="Number of frames to stack across time")
args = parser.parse_args()

SEED = args.seed
OBS_REPEAT = args.obs_repeat
OBS_NOISE = args.obs_noise
EXTRA_OBS_DIMS = args.extra_obs_dims
EXTRA_OBS_NOISE_STD = args.extra_obs_noise_std
FRAME_STACK = args.frame_stack
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
n_envs          = int(hp.get("n_envs", 1))
use_sde         = bool(hp.get("use_sde", False))
sde_sample_freq = int(hp.get("sde_sample_freq", -1))


policy_kwargs = {}
if "policy_kwargs" in hp:
    policy_kwargs = eval(
        hp["policy_kwargs"],
        {"__builtins__": None},
        {"nn": nn, "torch": torch, "dict": dict}
    )

# Construct a descriptive run name
run_name = (f"{ENV_NAME.lower()}-x{OBS_REPEAT}-seed{SEED}" + (f"-noise{OBS_NOISE}" if OBS_NOISE > 0 else ""))

USE_WANDB = os.environ.get("USE_WANDB", "1") != "0"
if USE_WANDB:
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
else:
    config = {
        **hp,
        "seed": SEED,
        "obs_repeat": OBS_REPEAT,
        "obs_noise": OBS_NOISE,
    }

# Save run metadata
with open(os.path.join(RUN_DIR, "parameters.yaml"), "w") as f:
    yaml.dump(dict(config), f)
with open(os.path.join(RUN_DIR, "command.txt"), "w") as f:
    f.write("python " + " ".join(sys.argv) + "\n")

# --- Environment creation using DummyVecEnv, VecNormalize, AddGaussianNoise ---
env_wrapper_path = hp.get("env_wrapper", None)
env_wrapper_cls = None
if env_wrapper_path:
    module_name, class_name = env_wrapper_path.rsplit(".", 1)
    env_wrapper_cls = getattr(importlib.import_module(module_name), class_name)

def make_env():
    def _thunk():
        try:
            try:
                env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)
            except TypeError:
                env = gym.make(ENV_NAME)
        except Exception as e:
            if isinstance(e, gym.error.NameNotFound):
                print(f"ERROR: Environment '{ENV_NAME}' does not exist or is not installed. Please check your environment name and dependencies.")
                sys.exit(1)
            else:
                raise
        env.reset(seed=SEED)
        env = Monitor(env)
        if OBS_REPEAT > 1:
            env = ObservationRepeater(env, repeat=OBS_REPEAT)
        # Apply env_wrapper if specified
        if env_wrapper_cls is not None:
            env = env_wrapper_cls(env) # TimeFeatureWrapper in AntBullet-v0
        return env
    return _thunk

# Use n_envs for parallel environments
raw_vec = DummyVecEnv([make_env() for _ in range(n_envs)])
if FRAME_STACK > 1:
    raw_vec = VecFrameStack(raw_vec, n_stack=FRAME_STACK)
vec_norm = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, training=True)

if OBS_NOISE > 0:
    vec_norm = AddGaussianNoise(vec_norm, sigma=OBS_NOISE)

# -------- add extra (unnormalised) noise dimensions ---------------------
if EXTRA_OBS_DIMS > 0:
    env = AddExtraObsDims(vec_norm,
                          extra_dims=EXTRA_OBS_DIMS,
                          std=EXTRA_OBS_NOISE_STD)
else:
    env = vec_norm

# Initialize PPO model, passing use_sde and sde_sample_freq
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
    use_sde=use_sde,
    sde_sample_freq=sde_sample_freq,
    verbose=1,
    tensorboard_log=os.path.join(RUN_DIR, "tensorboard"),
)

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path=RUN_DIR,
    name_prefix="checkpoint"
)

if USE_WANDB:
    wandb_callback = WandbCallback(
        gradient_save_freq=10,
        model_save_path=RUN_DIR,
        verbose=2,
        log="all",
    )
    callback = CallbackList([checkpoint_callback, wandb_callback])
else:
    callback = CallbackList([checkpoint_callback])

# Train the agent
model.learn(
    total_timesteps=total_timesteps,
    callback=callback
)


# Save VecNormalize even if it's wrapped
def find_vecnormalize(e):
    while hasattr(e, 'venv'):
        if isinstance(e, VecNormalize):
            return e
        e = e.venv
    return None

vec_to_save = find_vecnormalize(env)
if vec_to_save is not None:
    vecnorm_path = os.path.join(RUN_DIR, "vecnormalize.pkl")
    vec_to_save.save(vecnorm_path)
    print(f"VecNormalize stats saved to {vecnorm_path}")
else:
    print("Warning: VecNormalize not found in final env wrapper stack. Nothing saved.")

def unwrap_chain(e):
    """Return OUTER → INNER wrapper names covering both VecEnv and first worker."""
    chain = []
    def walk(obj):
        chain.append(obj.__class__.__name__)
        if hasattr(obj, "venv"):          # VecEnvWrapper
            walk(obj.venv)
        elif hasattr(obj, "env"):         # gym.Wrapper
            walk(obj.env)
        elif isinstance(obj, DummyVecEnv):
            walk(obj.envs[0])             # dive into first worker
    walk(e)
    return chain

wrapper_stack = unwrap_chain(env)          # env is the one given to PPO


env_stack = {
    "wrapper_stack": wrapper_stack,
    "parameters": {
        "obs_repeat": OBS_REPEAT,
        "obs_noise": OBS_NOISE,
        "extra_obs_dims": EXTRA_OBS_DIMS,
        "extra_obs_noise_std": EXTRA_OBS_NOISE_STD,
        "env_wrapper": env_wrapper_path,
        "frame_stack": FRAME_STACK,
    },
}

with open(os.path.join(RUN_DIR, "env_stack.json"), "w") as f:
    json.dump(env_stack, f, indent=2)
print(f"Environment wrapper stack saved to {os.path.join(RUN_DIR, 'env_stack.json')}")


# if USE_WANDB:
#     artifact = wandb.Artifact(name=run_name, type="model")
#     artifact.add_file(final_model_path)
#     wandb.log_artifact(artifact)
#     wandb.finish()

