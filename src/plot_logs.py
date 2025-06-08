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
    "pdf.fonttype": 42,  # For editable text in PDFs
})

# === LOG DIRECTORY (set here) ===
LOG_DIR = "../runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_20-extra_std_1.0/run1"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS = sorted([
    d for d in os.listdir(LOG_DIR)
    if os.path.isdir(os.path.join(LOG_DIR, d))
])

def get_event_file(run_folder):
    # Specifically look in run_folder/tensorboard/
    tb_dir = os.path.join(run_folder, "tensorboard")
    if os.path.isdir(tb_dir):
        event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
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

    for run, (tags, ea) in run_event_data.items():
        if tag not in tags:
            continue
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })
        sns.lineplot(data=df, x="step", y="value", label=f"Seed {run}", ax=ax)
        has_data = True

    if has_data:
        clean_tag = tag.replace("rollout/", "").replace("_", " ").title()
        ax.set_title(clean_tag)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(clean_tag)
        ax.legend(loc="best", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f"{tag.replace('/', '_')}.pdf")
        plt.savefig(filename, bbox_inches="tight")
        print(f"[✓] Saved: {filename}")
        plt.close()
    else:
        print(f"[!] Skipping {tag}: no data in any run.")
