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
import itertools
from tensorboard.backend.event_processing import event_accumulator

# ──────────────────────────── USER INPUT ────────────────────────────
# Map <nice label> → <path to run1/>
BASELINE_RUNS: Dict[str, str] = {
    "PPO 40 Dim Gaussian Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_40-extra_std_1.0-frames_4-policy_MlpPolicy/run1",
    # "PPO 20 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_20-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
    # "PPO 40 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-policy_MlpPolicy/run1",
}
ATTN_RUNS: Dict[str, str] = {
    "Attention 40 Dim Gaussian Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_gaussian-extra_dims_40-extra_std_1.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
    # "Attention 20 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_20-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
    # "Attention 40 Dim Uniform Noise": "/home/damlakonur/tum-adlr-01/tum-adlr-01/runs/LunarLanderContinuous-v3/LunarLanderContinuous-v3-x1-obs_noise_0.0-extra_obs_type_uniform-extra_dims_40-extra_std_1.0-mu_low-10.0-mu_high10.0-frames_4-attn_acttrue-attn_valtrue-attn_commonfalse-policy_FrameAttentionPolicy/run1",
}
TAG: str = "rollout/ep_rew_mean"   # TensorBoard scalar to plot
SAVE_DIR: str = "figures"
OUTPUT_FILE: str = os.path.join(SAVE_DIR, "multi_noise_comparison_Gaussian.png")
# ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
# Larger, bold fonts for poster readability
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.5,
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color cycle for noise levels
    NOISE_COLORS = {
        "10": "tab:green",
        "20": "tab:orange",
        "40": "tab:purple",
    }

    MARKER_BASELINE = "o"   # circle for PPO
    MARKER_ATTENTION = "s"  # square for attention
    LINESTYLE_BASELINE = "-"
    LINESTYLE_ATTENTION = "--"

    def detect_noise_dim(label: str) -> str:
        """Extract numeric noise-dimension token ('10', '20', '40', …) from label."""
        m = re.search(r"(\d+)\s*Dim", label)
        return m.group(1) if m else "unknown"

    # Plot baseline and attention runs interleaved so legend groups stay ordered.
    for mapping, marker, ls in ((BASELINE_RUNS, MARKER_BASELINE, LINESTYLE_BASELINE),
                                (ATTN_RUNS, MARKER_ATTENTION, LINESTYLE_ATTENTION)):
        for lbl, path in mapping.items():
            noise_key = detect_noise_dim(lbl)
            color = NOISE_COLORS.get(noise_key, "grey")
            try:
                steps, mean, std = _aggregate_seeds(path, TAG)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[!] {e}")
                continue
            ax.plot(steps, mean, label=lbl, color=color, marker=marker,
                    markevery=0.15, linewidth=1.8, linestyle=ls)

    ax.set_title("Attention vs PPO with 40 Dim Gaussian Noise")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (mean)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"[✓] Figure saved to {OUTPUT_FILE}")
