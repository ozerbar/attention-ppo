#!/usr/bin/env python3
"""
plot_seed0_ep_rew_mean.py

Plots the episode‑mean reward (TensorBoard tag: ``rollout/ep_rew_mean``)
for *seed 0* runs coming from **multiple experiment folders** and overlays
all curves in a single figure.

---
Usage
-----
1. Fill the ``RUN_PATHS`` dictionary with <legend‑label>: <path‑to‑folder>
   pairs.  Each *path* can point either directly to a ``seed0`` run folder
   or any ancestor directory that contains it.
2. Optionally tweak ``ANNOTATE_MAX`` or the global Matplotlib style.
3. Run the script: ``python plot_seed0_ep_rew_mean.py``. A PNG called
   ``seed0_ep_rew_mean.png`` will be written next to the script.

Requirements: ``tensorboard``, ``pandas``, ``matplotlib``, ``seaborn``.
"""
from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


# ───────────────────────── USER INPUT ──────────────────────────
# Map <legend label> ▶ <folder‑path>. Add as many as you like.
RUN_PATHS: Dict[str, str] = {
    "obsx1": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_0-extra_std_0.0/run1/",
    "obsx4": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-zoo-repeat4-noise0/run1/",
    "obsx16": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-zoo-repeat16-noise0/run1/",
    "obsx32": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-zoo-repeat32-noise0/run1/",
    "10 noise dim, std=0.5": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_10-extra_std_0.5/run1",
    "20 noise dim, std=1.0": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_20-extra_std_1.0/run1",
    "40 noise dim, std=0.5": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_40-extra_std_0.5/run1",
    "40 noise dim, std=1.0": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_40-extra_std_1.0/run1",
    "80 noise dim, std=2.0": "/home/ozerbar/tum-adlr-01/runs/AntBulletEnv-v0/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_80-extra_std_2.0/run1",


}

TAG: str = "rollout/ep_rew_mean"  # TensorBoard scalar to plot.
OUTPUT_FILE: str = "seed0_ep_rew_mean.png"  # Output figure filename.
ANNOTATE_MAX: bool = True  # Draw a dashed line and text at global max.


SAVE_DIR: str = "/home/ozerbar/tum-adlr-01/figures"  # <-- your desired directory
OUTPUT_FILE: str = os.path.join(SAVE_DIR, "seed0_ep_rew_mean.png")
# ───────────────────────────────────────────────────────────────

# 🔧 Global plot appearance
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def _find_event_file(root: str | os.PathLike) -> str | None:
    """Return the *first* TensorBoard event file inside *root*.

    Preference order:
        1. Any file that lives inside a directory whose name ends with
           ``seed0`` or ``seed_0`` or ``seed-0``.
        2. Otherwise the first event file found during the recursive scan.
    """
    root = Path(root).expanduser().resolve()

    # Allow users to pass the event file path directly → no search needed.
    if root.is_file() and root.name.startswith("events.out.tfevents"):
        return str(root)

    # Recursively collect all event files under *root*.
    pattern = str(root / "**" / "events.out.tfevents.*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None

    # Prefer those whose *parent* dir name advertises seed 0.
    seed0_regex = re.compile(r"seed[-_]?0$")
    seed0_files = [f for f in candidates if seed0_regex.search(Path(f).parent.name)]
    return (seed0_files or candidates)[0]  # first match is fine.


def _load_scalar_df(event_file: str, tag: str) -> pd.DataFrame:
    """Load *tag* from *event_file* → DataFrame(step, value)."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()  # parse the file
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame()
    events = ea.Scalars(tag)
    return pd.DataFrame({
        "step": [e.step for e in events],
        "value": [e.value for e in events],
    })


# ══════════════════ MAIN – collect data and plot ═══════════════
if __name__ == "__main__":
    if not RUN_PATHS:
        raise SystemExit("✗ RUN_PATHS is empty. Please specify <label>: <path> pairs.")

    fig, ax = plt.subplots(figsize=(10, 5))
    all_values = []

    for label, path in RUN_PATHS.items():
        event_file = _find_event_file(path)
        if event_file is None:
            print(f"[!] No TensorBoard file found for '{label}' under '{path}'. Skipping.")
            continue

        df = _load_scalar_df(event_file, TAG)
        if df.empty:
            print(f"[!] Tag '{TAG}' missing in '{event_file}'. Skipping.")
            continue

        sns.lineplot(
            data=df, x="step", y="value",
            ax=ax, estimator=None, label=label,
        )
        all_values.extend(df["value"].tolist())
        print(f"[✓] Added '{label}' from {event_file}")

    # Aesthetics
    ax.set_title("Episode Mean Reward (seed 0)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Dynamic y‑limits with a bit of head‑room.
    if all_values:
        ymin, ymax = min(all_values), max(all_values)
        padding = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_ylim(ymin - padding, ymax + padding)

    # Annotate global maximum (across all runs)
    if ANNOTATE_MAX and all_values:
        global_max = max(all_values)
        ax.axhline(global_max, linestyle="--", linewidth=1.5)
        ax.text(
            x=ax.get_xlim()[1], y=global_max,
            s=f"Max: {global_max:.2f}", ha="right", va="bottom", fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )

    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"\n[✓] Figure {OUTPUT_FILE} saved to {SAVE_DIR}\n")

    # plt.tight_layout()
    # plt.savefig(OUTPUT_FILE, dpi=150)
    # print(f"\n[✓] Figure saved to {OUTPUT_FILE}\n")
