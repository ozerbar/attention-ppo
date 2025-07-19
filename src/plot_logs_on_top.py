#!/usr/bin/env python3
"""
plot_ep_rew_mean_all_seeds.py

Plots the episode‑mean reward (TensorBoard tag: ``rollout/ep_rew_mean``)
for all seed runs in each experiment folder and overlays all curves per group
in a single color.

Each group (e.g. "obsx4") aggregates runs like seed0, seed1, seed2.

Requirements: ``tensorboard``, ``pandas``, ``matplotlib``, ``seaborn``.
"""

from __future__ import annotations

import os
import glob
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


# ───────────────────────── USER INPUT ──────────────────────────
RUN_PATHS: Dict[str, str] = {
    "obsx1": "/home/damlakonur/tum-adlr-01/runs/Pusher-v5/obsx1/",
    "obsx2": "/home/damlakonur/tum-adlr-01/runs/Pusher-v5/obsx2/",
    "obsx4": "/home/damlakonur/tum-adlr-01/runs/Pusher-v5/obsx4/",
    "obsx16": "/home/damlakonur/tum-adlr-01/runs/Pusher-v5/obsx16/",
}

TAG: str = "rollout/ep_rew_mean"
ANNOTATE_MAX: bool = True
SAVE_DIR: str = "/home/damlakonur/tum-adlr-01/figures"
OUTPUT_FILE: str = os.path.join(SAVE_DIR, "ep_rew_mean_all_seeds.png")

# ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def find_event_files(root: str | os.PathLike) -> List[str]:
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

def find_event_files2(root: str | os.PathLike) -> List[str]:
    """
    Recursively find all TensorBoard event files under seed*/**/ folders inside a given run1/.
    Works with structures like:
      - seed0/ppo/.../tb_logs/
      - seed0/runs/.../
      - seed0/tensorboard/.../
    """
    root = Path(root).expanduser().resolve()
    pattern = str(root / "**" / "events.out.tfevents.*")
    all_event_files = glob.glob(pattern, recursive=True)

    # Keep only those under a path segment matching seed0, seed1, ...
    seed_event_files = []
    for f in all_event_files:
        parts = Path(f).parts
        if any(re.fullmatch(r"seed\d+", p) for p in parts):
            seed_event_files.append(str(f))

    return sorted(seed_event_files)

def load_scalar_df(event_file: str, tag: str) -> pd.DataFrame:
    """Load *tag* from *event_file* → DataFrame(step, value)."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
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

    # Generate consistent color palette for all experiments
    palette = sns.color_palette("Accent", n_colors=len(RUN_PATHS)+6)

    for i, (label, root_path) in enumerate(RUN_PATHS.items()):
        event_files = find_event_files2(root_path)
        if not event_files:
            print(f"[!] No event files found under {root_path}")
            continue

        dfs = []
        for event_file in event_files:
            df = load_scalar_df(event_file, TAG)
            if df.empty:
                print(f"[!] No data for tag '{TAG}' in {event_file}")
                continue

            seed_name = Path(event_file).parts[
                list(Path(event_file).parts).index("run1") + 1
            ]  # e.g. seed0, seed1, seed2

            df["run"] = seed_name
            dfs.append(df)
            all_values.extend(df["value"].tolist())

        if not dfs:
            continue

        combined_df = pd.concat(dfs)
        color = palette[i+2]
        first = True  # flag to add only one legend entry per group

        for run_name, run_df in combined_df.groupby("run"):
            sns.lineplot(
                data=run_df, x="step", y="value",
                label=label if first else None,  # only first seed gets a label
                ax=ax, color=color, linewidth=1.3
            )
            first = False  # disable after first

        print(f"[✓] Plotted {len(dfs)} runs for '{label}'")

    ax.set_title("Episode Mean Reward (All Seeds)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if all_values:
        ymin, ymax = min(all_values), max(all_values)
        padding = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_ylim(ymin - padding, ymax + padding)

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
    print(f"\n[✓] Figure saved to {OUTPUT_FILE}\n")