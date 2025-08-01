# Learning What Matters: A Problem in Robotic Reinforcement Learning

> **Technical University of Munich – Advanced Deep Learning for Robotics (SS 24/25)**

This repository contains the code for the course project *“Learning What Matters: A Problem in Robotic Reinforcement Learning.”*  
We study how attention mechanisms can help an agent *ignore noisy dimensions* in its observations.  Concretely, we extend the Proximal Policy Optimisation (PPO) algorithm with a family of attention-based policies and compare them to the default multilayer-perceptron (MLP) baseline on classic continuous-control tasks.

---

## Table of contents
1. [Project structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   * [Quick start with `run_all.sh`](#quick-start-with-run_allsh)
   * [Manual run via `train.py`](#manual-run-via-trainpy)
   * [Command-line flags](#command-line-flags)
4. [Attention policy variants](#attention-policy-variants)
5. [Reproducing the paper results](#reproducing-the-paper-results)
6. [Acknowledgements](#acknowledgements)

---

## Project structure
```
├── environment.yml            # Conda environment (direct dependencies)
├── scripts/
│   ├── run_all.sh             # Launches 3 seeds with custom parameters
│   └── run_all_zoo.sh         # Same but uses rl-baselines3-zoo hyper-params
├── hyperparams/               # YAML files with PPO hyper-parameters per env
├── src/
│   ├── train.py               # Main training entry-point
│   ├── custom_policy.py       # Attention policy implementations
│   ├── observation_wrappers.py# Extra observation / noise wrappers
│   └── ...
└── runs/                      # Output will be written here (ignored by git)
```

*Every path in this README is relative to the repository root.*

---

## Installation
All direct dependencies are listed in `environment.yml`.

```bash
# 1. create the environment (≈ 3 min with mamba)
conda env create -f environment.yml

# 2. activate it
conda activate rl-env

# 3. (optional) verify
python -m pip list | grep stable-baselines3
```

---

## Usage

### Quick start with `run_all.sh`
The easiest way to reproduce our main experiment is to run:

```bash
sh scripts/run_all.sh
```

The script will:
1. Append the repo root to `PYTHONPATH` so that `src/` is importable.
2. Create a structured output directory under `runs/`.
3. Launch **three parallel seeds** (0, 1, 2) of `src/train.py` with the parameters you configure at the *top* of the script.

Edit the first ~30 lines to choose:
* the **environment** (`ENV_NAME="LunarLanderContinuous-v3"`)
* amount of **observation repetition / Gaussian noise**
* **extra noise dimensions** to inject (`EXTRA_OBS_DIMS`, `EXTRA_OBS_TYPE`, `EXTRA_OBS_NOISE_STD`, `MU_LOW`, `MU_HIGH`)
* **frame stacking** (`FRAME_STACK`)
* **attention flags** (`ATTN_ACT`, `ATTN_VAL`, `ATTN_COMMON`, `ATTN_DIRECT_OVERRIDE`)
* whether to log to **Weights & Biases** (`USE_WANDB`)

The script automatically selects the right policy class and hyper-parameter file, builds the command line, and organises the outputs like
```
runs/
└─ LunarLanderContinuous-v3/
   └─ LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-.../   # hyper-param signature
      └─ run1/           # one call of run_all.sh = one "batch" directory
         ├─ seed0/
         ├─ seed1/
         └─ seed2/
```

### Command-line flags
Below is the full list of flags parsed in [`src/train.py`](src/train.py).  Defaults are shown in brackets.

| Flag | Description |
|------|-------------|
| `--seed <int>` **(required)** | Random seed for reproducibility and for the output directory name.
| `--obs_repeat <int> [1]` | Repeat each observation `n` times before the agent sees the next state.  Useful for frame-skip style augmentation.
| `--obs_noise <float> [0.0]` | Standard deviation of *Gaussian* noise added to **all** original observation dimensions.
| `--extra_obs_type <str> ["linear"]` | How to generate *extra* observation dimensions. Choices: `gaussian`, `linear` (ramp), `uniform` (bounded noise).
| `--extra_obs_dims <int> [0]` | Number of extra noise dimensions to append to the observation vector.
| `--extra_obs_noise_std <float> [0.0]` | Std-dev of the extra dimensions when `extra_obs_type="gaussian"`.
| `--mu_low <float> [-1.0]` & `--mu_high <float> [1.0]` | Lower & upper bounds for *uniform* extra noise.
| `--frame_stack <int> [1]` | Temporal stacking of observations (works with `VecFrameStack`).
| `--conf-file <path>` | YAML file with PPO hyper-parameters.  If omitted, defaults to `hyperparams/<ENV_NAME>.yml`.
| `--policy <str> ["MlpPolicy"]` | Which policy class to instantiate.  See [Attention policy variants](#attention-policy-variants).
| `--attn_act` | Enable attention blocks in the **actor** network (only relevant for attention policies).
| `--attn_val` | Enable attention blocks in the **critic** network.
| `--attn_common` | Use a *shared* attention block for both actor & critic.

Environment variables recognised at runtime:
* `ENV_NAME` (string, **required**) – Gymnasium environment ID.
* `RUN_BATCH_DIR` (path, **required**) – directory where all outputs are written.
* `USE_WANDB` (`"1"`|`"0"`, default `"1"`) – toggle Weights & Biases logging.

---

## Attention policy variants
All policies are registered in `train.py` as
```python
POLICY_REGISTRY = {
    "MlpPolicy":                 "MlpPolicy",          # SB3 default MLP
    "AttentionPolicy":          AttentionPolicy,
    "SelectiveAttentionPolicy": SelectiveAttentionPolicy,
    "AttentionDirectOverridePolicy": AttentionDirectOverridePolicy,
    "MediumAttentionPolicy":    MediumAttentionPolicy,
    "FrameAttentionPolicy":     FrameAttentionPolicy,
}
```

| Policy | Idea | Typical use |
|--------|------|-------------|
| `MlpPolicy` | Plain fully-connected MLP (baseline) | Control comparison.
| `AttentionPolicy` | Adds attention blocks **either** in actor (--attn_act) or critic (--attn_val) | Ablation.
| `SelectiveAttentionPolicy` | Allows separate or shared attention; can learn to *mask* irrelevant dims | Ablation.
| `AttentionDirectOverridePolicy` | Attention output directly *overrides* selected features | Hard gating experiment.
| `MediumAttentionPolicy` | Smaller hidden size; fewer parameters | Parameter-controlled study.
| `FrameAttentionPolicy` | Self-attention over stacked frames (temporal) | Main Method.

See `src/custom_policy.py` for implementation details.

---

## Reproducing the paper results
1. Clone the repo and install the environment as described above.
2. Choose the desired environment and noise configuration in `scripts/run_all.sh`.
3. Execute the script and wait until all three seeds finish.  Checkpoints, TensorBoard logs, and configuration files will appear under `runs/…`.
4. Visualise training curves with
   ```bash
   tensorboard --logdir runs
   ```
5. If `USE_WANDB=1`, interactive dashboards are also uploaded to our public WandB project `adlr-01/*`.

---

## Acknowledgements
This project builds on
* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
* [PyBullet environments](https://github.com/erwincoumans/pybullet)
* The *rl-baselines3-zoo* hyper-parameter collection

---