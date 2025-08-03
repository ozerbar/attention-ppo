import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import rcParams
import matplotlib.cm as cm

PRINT_MEAN_STD = True
SAVE_DIR = "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/figures"
PLOT_TITLE = "AntBullet: PPO vs Attention PPO Gaussian Noise"


EXPERIMENTS = {
    "Standard PPO - 40D": "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_40-extra_std_1.0-frames_4-policy_MlpPolicy/run2",
    "Attention PPO - 40D": "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_40-extra_std_1.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
    "Standard PPO - 100D": "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_100-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
    "Attention PPO - 100D": "/home/ozerbar/Downloads/tum_adlr/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_100-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
}

# ======================================================================================

def get_distinct_colors(n):
    colormap = cm.get_cmap('tab10' if n <= 10 else 'tab20')
    return [colormap(i) for i in range(n)]

def title_to_filename(title: str) -> str:
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s\-+]', '', title)
    title = title.replace('+', 'plus').replace('-', 'minus')
    title = re.sub(r'\s+', '_', title.strip())
    return title + ".png"

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

def assign_color(label):
    if "Attention" in label:
        return "blue"
    elif "Standard" in label:
        return "red"
    elif "Baseline" in label:
        return "green"
    else:
        return "gray"

def plot_comparison(experiments, title, save_dir, tag="rollout/ep_rew_mean"):
    rcParams.update({
        'font.size': 22,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'legend.frameon': True,
        'lines.linewidth': 3.0,
        'figure.dpi': 100
    })

    plt.figure(figsize=(14, 8))
    colors = get_distinct_colors(len(experiments))
    for idx, (label, log_path) in enumerate(experiments.items()):
        steps, vals = collect_all_runs(log_path, tag)
        if vals.size > 0:
            mean = np.mean(vals, axis=0)
            std = np.std(vals, axis=0)
            color = colors[idx]
            plt.plot(steps, mean, label=label, color=color)
            if PRINT_MEAN_STD:
                plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward (Mean ± Std)" if PRINT_MEAN_STD else "Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title_to_filename(title))
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

# ======================================================================================

def main():
    plot_comparison(EXPERIMENTS, PLOT_TITLE, SAVE_DIR)

if __name__ == "__main__":
    main()
