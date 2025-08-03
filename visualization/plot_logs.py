import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# === GLOBAL PLOT CONFIG ===
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

})

# === LOG DIRECTORY (set here) ===

# RELATIVE_DIR = "runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_0-extra_std_0.0-frames_10"

# LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", RELATIVE_DIR, "run1/"))

LOG_DIR  = "/home/damlakonur/tum-adlr-01/AntBulletEnv-v0/AntBulletEnv-v0-zoo-repeat32-noise0/run1/"


ANNOTATE_MAX = True  # Whether to annotate the max value on "rollout/ep_rew_mean"



OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_event_files(root: str | os.PathLike):
    """
    Recursively find all TensorBoard event files under given run1/ folder,
    grouping by seed0, seed1, seed2. Works with mixed nesting structures.
    """
    root = Path(root).expanduser().resolve()
    pattern = str(root / "seed*" / "**" / "events.out.tfevents.*")
    candidates = glob.glob(pattern, recursive=True)

    event_files = []
    for f in candidates:
        path = Path(f)
        # Only accept if part of a valid seed folder (seed0, seed1, seed2, etc.)
        if any(re.match(r"seed[0-9]+$", part) for part in path.parts):
            event_files.append(str(path))

    return sorted(event_files)


def load_scalars(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    return ea.Tags().get("scalars", []), ea

# === COLLECT ALL RUN FOLDERS THAT CONTAIN TENSORBOARD FILES ===
RUNS = []
run_event_data = {}
all_tags = set()

candidate_dirs = [os.path.join(LOG_DIR, d) for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]

for run_dir in candidate_dirs:
    event_file = find_event_files(run_dir)
    if not event_file:
        continue

    run_name = os.path.basename(run_dir)

    # Try to extract a seed number intelligently
    if "seed" in run_name.lower():
        seed_label = run_name.lower()
    elif "ppo" in run_name.lower():
        seed_label = run_name.lower()
    else:
        # fallback to full folder name
        seed_label = run_name

    tags, ea = load_scalars(event_file)
    run_event_data[seed_label] = (tags, ea)
    all_tags.update(tags)

# === PLOT EACH TAG ===
for tag in sorted(all_tags):
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False
    all_values = []
    max_points = []  # To collect (step, value) for max point annotations

    for run, (tags, ea) in run_event_data.items():
        if tag not in tags:
            continue
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })
        all_values.extend(df["value"])

        match = re.search(r'\d+', run)
        if match:
            seed_number = match.group(0)
            label = f"Seed = {seed_number}"
        else:
            label = f"Run: {run}"  # fallback if no digits

        sns.lineplot(data=df, x="step", y="value", label=label, ax=ax, estimator=None)
        has_data = True

        if tag == "rollout/ep_rew_mean" or "eval/mean_reward" and not df.empty:
            idxmax = df["value"].idxmax()
            max_step = df.loc[idxmax, "step"]
            max_value = df.loc[idxmax, "value"]
            max_points.append((max_step, max_value))

    if has_data:
        # Auto y-axis limits with padding
        if all_values:
            ymin = min(all_values)
            ymax = max(all_values)
            padding = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - padding, ymax + padding)

        # Annotate max points on ep_rew_mean
        if ANNOTATE_MAX and tag == "rollout/ep_rew_mean" and all_values:
            max_val = max(all_values)
            ax.axhline(y=max_val, color="red", linestyle="--", linewidth=1.5)
            ax.text(
                x=ax.get_xlim()[1], y=max_val, s=f"Max: {max_val:.2f}",
                color="red", va="bottom", ha="right", fontsize=12, fontweight="bold",
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0)
            )

        clean_tag = tag.replace("rollout/", "").replace("_", " ").title()
        ax.set_title(clean_tag)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(clean_tag)
        ax.legend(loc="best", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f"{tag.replace('/', '_')}.png")
        plt.savefig(filename, bbox_inches="tight")
        print(f"[✓] Saved: {filename}")
        plt.close()
    else:
        print(f"[!] Skipping {tag}: no data in any run.")