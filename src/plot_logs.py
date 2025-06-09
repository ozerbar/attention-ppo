import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

RELATIVE_DIR = "runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_0-extra_std_0.0-frames_10"

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", RELATIVE_DIR, "run1/"))

OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS = sorted([
    d for d in os.listdir(LOG_DIR)
    if os.path.isdir(os.path.join(LOG_DIR, d))
])

def get_event_file(run_folder):
    # Search all event files recursively under the run_folder
    event_files = glob.glob(os.path.join(run_folder, "**", "events.out.tfevents.*"), recursive=True)
    if event_files:
        return event_files[0]
    return None



def load_scalars(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    return ea.Tags().get("scalars", []), ea

# === GATHER ALL TAGS ===
all_tags = set()
run_event_data = {}

for run in RUNS:
    run_path = os.path.join(LOG_DIR, run)
    event_file = get_event_file(run_path)
    if not event_file:
        print(f"[!] No event file found in: {run_path}")
        continue

    tags, ea = load_scalars(event_file)
    run_event_data[run] = (tags, ea)
    all_tags.update(tags)

# === PLOT EACH TAG ===
for tag in sorted(all_tags):
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False
    all_values = []

    for run, (tags, ea) in run_event_data.items():
        if tag not in tags:
            continue
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })
        all_values.extend(df["value"])  # collect for y-limits
        seed_number = ''.join(filter(str.isdigit, run))  # extracts the number from 'seed0', 'seed1', etc.
        sns.lineplot(data=df, x="step", y="value", label=f"Seed = {seed_number}", ax=ax, estimator=None)
        has_data = True

    if has_data:
        # Auto y-axis limits with small padding
        if all_values:
            ymin = min(all_values)
            ymax = max(all_values)
            padding = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - padding, ymax + padding)

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
