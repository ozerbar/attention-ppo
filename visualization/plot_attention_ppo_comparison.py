import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import rcParams

# =============================================================================
PRINT_MEAN_STD = True # Set this to True for mean ± std plot, False for individual seed plots

SAVE_DIR = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/figures"

# PLOT_TITLE = "40 Extra Dim Uniform Noise -10/+10 Range"
# ATTN_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1"
# PPO_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run2"

# PLOT_TITLE = "40 Extra Dim Ramp Noise with 0.001 Slope"
# ATTN_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_linear-extra_dims_40-extra_std_0.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run3"
# PPO_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_linear-extra_dims_40-extra_std_0.0-frames_4-policy_MlpPolicy/run3"

PLOT_TITLE = "100 Extra Dim Gaussian Noise with std=1.0"
ATTN_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_100-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1"
PPO_PATH = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_100-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1"



# =============================================================================

def title_to_filename(title: str) -> str:
    # Lowercase, remove special chars, replace spaces with underscores
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s\-+]', '', title)  # remove non-alphanumerics except +/-
    title = title.replace('+', 'plus').replace('-', 'minus')
    title = re.sub(r'\s+', '_', title.strip())  # spaces to underscores
    return title + ".png"
filename = title_to_filename(PLOT_TITLE)
SAVE_PATH = os.path.join(SAVE_DIR, filename)

def load_scalar_from_event_file(event_file, tag='rollout/ep_rew_mean'):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        print(f"Tag {tag} not found in {event_file}")
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

def find_event_file(seed_path):
    tb_dir = os.path.join(seed_path, "tensorboard", "PPO_1")
    if not os.path.exists(tb_dir):
        return None
    for fname in os.listdir(tb_dir):
        if fname.startswith("events.out.tfevents"):
            return os.path.join(tb_dir, fname)
    return None

def collect_all_runs(base_path, tag):
    seeds = ["seed0", "seed1", "seed2"]
    all_vals = []
    all_steps = None
    min_len = None

    # First pass: find common min length
    for seed in seeds:
        event_file = find_event_file(os.path.join(base_path, seed))
        if event_file:
            steps, vals = load_scalar_from_event_file(event_file, tag)
            if len(vals) > 0:
                if min_len is None:
                    min_len = len(vals)
                    all_steps = steps
                else:
                    min_len = min(min_len, len(vals), len(steps))

    # Second pass: collect only trimmed arrays
    for seed in seeds:
        event_file = find_event_file(os.path.join(base_path, seed))
        if event_file:
            steps, vals = load_scalar_from_event_file(event_file, tag)
            if len(vals) >= min_len:
                all_vals.append(vals[:min_len])
    if all_steps is not None and min_len is not None:
        return all_steps[:min_len], np.stack(all_vals)
    else:
        return [], np.array([])

def plot_attention_vs_standard(attn_path, ppo_path, save_path=SAVE_PATH, tag="rollout/ep_rew_mean"):
    # Style
    rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'legend.title_fontsize': 22,
    'legend.frameon': True,
    'lines.linewidth': 3.5,
    'figure.dpi': 100
})

    plt.figure(figsize=(12, 7))

    # PPO
    ppo_steps, ppo_vals = collect_all_runs(ppo_path, tag)
    if ppo_vals.size > 0:
        if PRINT_MEAN_STD:
            mean = np.mean(ppo_vals, axis=0)
            std = np.std(ppo_vals, axis=0)
            plt.plot(ppo_steps, mean, color="red", label="Standard PPO")
            plt.fill_between(ppo_steps, mean - std, mean + std, color="red", alpha=0.2)
        else:
            for i, vals in enumerate(ppo_vals):
                plt.plot(ppo_steps, vals, color="red", alpha=0.6, label="Standard PPO" if i == 0 else None)

    # Attention PPO
    attn_steps, attn_vals = collect_all_runs(attn_path, tag)
    if attn_vals.size > 0:
        if PRINT_MEAN_STD:
            mean = np.mean(attn_vals, axis=0)
            std = np.std(attn_vals, axis=0)
            plt.plot(attn_steps, mean, color="blue", label="Attention PPO")
            plt.fill_between(attn_steps, mean - std, mean + std, color="blue", alpha=0.2)
        else:
            for i, vals in enumerate(attn_vals):
                plt.plot(attn_steps, vals, color="blue", alpha=0.6, label="Attention PPO" if i == 0 else None)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward (Mean)")
    plt.title(PLOT_TITLE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

def main():
    plot_attention_vs_standard(ATTN_PATH, PPO_PATH)

if __name__ == "__main__":
    main()
