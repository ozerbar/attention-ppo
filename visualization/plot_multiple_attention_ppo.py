#!/usr/bin/env python3
"""
plot_multiple_attention_ppo.py

Edit the `BASELINE_RUNS` and `ATTN_RUNS` dictionaries so that their keys are
human-readable labels (will be shown in the legend) and their values point to
the corresponding **run1** directory produced by `scripts/run_all.sh`.

Example directory layout expected:
    runs/AntBulletEnv-v0/AntBulletEnv-v0-…-policy_MlpPolicy/run1/seed0/
                                                         └─ tensorboard/PPO_1/events…
The script is robust to nested TB folders thanks to a recursive search.
"""
from __future__ import annotations

import os
import glob
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator

# ──────────────────────────── USER INPUT ────────────────────────────
# Map <nice label> → <path to run1/>
BASELINE_RUNS: Dict[str, str] = {
    "PPO 10 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_10-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
    "PPO 20 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_20-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
    "PPO 40 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
}
ATTN_RUNS: Dict[str, str] = {
    "Attention 10 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_10-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
    "Attention 20 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_20-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
    "Attention 40 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
}
TAG: str = "rollout/ep_rew_mean"   # TensorBoard scalar to plot
SAVE_DIR: str = "figures"
OUTPUT_FILE: str = os.path.join(SAVE_DIR, "multi_noise_comparison_uniform.png")
# ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
# Shrink font sizes to fit 5×3 inch canvas
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
})


def _find_event_files(run1_dir: str | os.PathLike) -> List[str]:
    """Return list of all *events.out.tfevents* files under seed*/**/ inside run1_dir."""
    pattern = str(Path(run1_dir).expanduser().resolve() / "seed*" / "**" / "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def _load_scalar_df(event_file: str, tag: str) -> pd.DataFrame:
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame()
    events = ea.Scalars(tag)
    return pd.DataFrame({
        "step": [e.step for e in events],
        "value": [e.value for e in events],
    })


def _aggregate_seeds(run1_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, std) across seeds.
    Steps are clipped to the shortest run so arrays align."""
    tables: List[pd.Series] = []
    steps_ref: np.ndarray | None = None
    min_len: int | None = None

    for ev_file in _find_event_files(run1_dir):
        df = _load_scalar_df(ev_file, tag)
        if df.empty:
            continue
        if min_len is None:
            min_len = len(df)
            steps_ref = df["step"].to_numpy()
        min_len = min(min_len, len(df))
    if min_len is None:
        raise FileNotFoundError(f"No scalar '{tag}' found under {run1_dir}")

    for ev_file in _find_event_files(run1_dir):
        df = _load_scalar_df(ev_file, tag)
        if df.empty or len(df) < min_len:
            continue
        tables.append(df["value"].to_numpy()[:min_len])

    if not tables:
        raise RuntimeError(f"Found event files but none contained tag '{tag}' under {run1_dir}")

    values = np.stack(tables)                # shape (num_seeds, min_len)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return steps_ref[:min_len], mean, std


def _plot_group(run_map: Dict[str, str], color: str, ax: plt.Axes):
    for label, path in run_map.items():
        try:
            steps, mean, std = _aggregate_seeds(path, TAG)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[!] {e}")
            continue
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


if __name__ == "__main__":
    if not BASELINE_RUNS or not ATTN_RUNS:
        raise SystemExit("✗ Please populate BASELINE_RUNS and ATTN_RUNS in the script header.")

    FIGSIZE = (3.6, 2.4)   # further reduced canvas size
    DPI = 300          # high resolution for poster printing
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Color encodes *model*
    COLOR_PPO = "purple"
    COLOR_ATTENTION = "orange"

    # Linestyle encodes *noise dimension*
    NOISE_STYLES = {
        "10": ":",   # dotted
        "20": "-",    # solid
        "40": "--",   # dashed
    }
    MARKER_BASELINE = "o"   # circle for PPO
    MARKER_ATTENTION = "s"  # square for Attention

    def detect_noise_dim(label: str) -> str:
        """Extract numeric noise-dimension token ('10', '20', '40', …) from label."""
        m = re.search(r"(\d+)\s*Dim", label)
        return m.group(1) if m else "unknown"

    # Plot PPO (purple) and Attention (orange)
    for mapping, marker, model_color in ((BASELINE_RUNS, MARKER_BASELINE, COLOR_PPO),
                                         (ATTN_RUNS, MARKER_ATTENTION, COLOR_ATTENTION)):
        for lbl, path in mapping.items():
            noise_key = detect_noise_dim(lbl)
            ls = NOISE_STYLES.get(noise_key, "-")
            color = model_color
            try:
                steps, mean, std = _aggregate_seeds(path, TAG)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[!] {e}")
                continue
            ax.plot(steps, mean, color=color, marker=marker,
                    markevery=0.15, linewidth=1.8, linestyle=ls, label="_nolegend_")

    # Restore original title
    ax.set_title("PPO vs Attention – Uniform Noise", weight="bold")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (mean)")

    # ── custom legend: color → model, linestyle → noise dim ─────────────
    legend_handles = [
        Line2D([0], [0], color=COLOR_PPO, lw=2, marker=MARKER_BASELINE, label="PPO"),
        Line2D([0], [0], color=COLOR_ATTENTION, lw=2, marker=MARKER_ATTENTION, label="Attention"),
        Line2D([0], [0], color="black", lw=2, linestyle=NOISE_STYLES["10"], label="10 dim"),
        Line2D([0], [0], color="black", lw=2, linestyle=NOISE_STYLES["20"], label="20 dim"),
        Line2D([0], [0], color="black", lw=2, linestyle=NOISE_STYLES["40"], label="40 dim"),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=1, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[✓] Figure saved to {OUTPUT_FILE}")
