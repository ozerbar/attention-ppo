from __future__ import annotations
import gymnasium as gym  # type: ignore
import pybullet_envs_gymnasium  # noqa: F401  (registers AntBullet‑v0)
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
import matplotlib.pyplot as plt  # Add this import



###############################################################################
############################## Constants ######################################
#
NOISE_STD = 0.3  # Standard deviation of the Gaussian noise to be added 
#
###############################################################################

###############################################################################
# 1. Environment factory ######################################################
###############################################################################

def make_env(seed: int | None = None):
    """Return a fresh AntBullet‑v0 env (without any wrappers)."""
    env = gym.make("AntBulletEnv-v0", render_mode=None)
    if seed is not None:
        env.reset(seed=seed)
    return env

###############################################################################
# 2. Noise wrapper (AFTER normalisation!) #####################################
###############################################################################

class AddGaussianNoise(VecEnvWrapper):
    def __init__(self, venv, sigma: float = 0.0, seed: int | None = None):
        super().__init__(venv)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _add_noise(self, obs: np.ndarray) -> np.ndarray:  # noqa: D401 (simple)
        """Return *obs + N(0, sigma)* with the same shape as *obs*."""
        noise = self.rng.normal(0.0, self.sigma, size=obs.shape)
        return obs + noise

    # ---------------------------------------------------------------------
    # VecEnv‑required overrides
    # ---------------------------------------------------------------------
    def reset(self, **kwargs):  # noqa: D401 (gym signature)
        obs = self.venv.reset(**kwargs)
        return self._add_noise(obs)

    def step_wait(self):  # type: ignore[override]
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._add_noise(obs), rewards, dones, infos

###############################################################################
# 3. Utility: warm‑up VecNormalize stats ######################################
###############################################################################

def warm_up(vec_env: VecNormalize, steps: int = 10_000) -> None:
    # Make sure we start from a defined state so the very first observation is
    # included in the running stats.
    vec_env.reset()

    for _ in range(steps):
        # VecEnv expects a list of actions, one per parallel environment.
        action = [vec_env.action_space.sample()]
        _, _, dones, _ = vec_env.step(action)
        # `dones` is a NumPy array with length = n_envs. Use np.any for clarity.
        if np.any(dones):
            vec_env.reset()


def sample_noisy_stats(env, n_steps=50_000):
    buf = []
    obs = env.reset()
    for _ in range(n_steps):
        # Dummy random action; wrap in list because DummyVecEnv expects a batch
        action = [env.action_space.sample()]
        obs, _, dones, _ = env.step(action)
        buf.append(obs[0])          # unwrap the single-env batch
        if np.any(dones):
            env.reset()

    data = np.asarray(buf)          # shape (n_steps, obs_dim)
    return data, data.mean(axis=0), data.var(axis=0)  # changed std to var



###############################################################################
# 4. Main script ##############################################################
###############################################################################

def main():  # noqa: D401 (entry point)
    # ------------------------------------------------------------------
    # Build the wrapper stack: raw env -> VecNormalize -> AddNoise
    # ------------------------------------------------------------------
    raw_vec = DummyVecEnv([make_env])
    vec_norm = VecNormalize(raw_vec, norm_obs=True, norm_reward=False, training=True)


    warm_up(vec_norm, steps=100_000)  # increased warm‑up to better stabilise running mean/var

    # # Stop updating stats (optional). Comment out if you prefer on‑going updates.
    # vec_norm.training = False


    print("-------------------------------------------------------------------")

    # Save stats for comparison
    mean_before = vec_norm.obs_rms.mean.copy()
    var_before = vec_norm.obs_rms.var.copy()
    # print(f"Mean vec_norm before noise: {mean_before}")
    # print(f"Var vec_norm before noise: {var_before}")

    noisy_env = AddGaussianNoise(vec_norm, sigma=NOISE_STD)

    # # Interact with the noisy environment
    # for _ in range(100000):
    #     a = [noisy_env.action_space.sample()]
    #     noisy_env.step(a)

    # print("-------------------------------------------------------------------")

    # # Interact with the not noisy environment
    # for _ in range(50_000):
    #     a = [vec_norm.action_space.sample()]
    #     noisy_env.step(a)

    # print("-------------------------------------------------------------------")



    # --- Sample statistics for plotting ---
    clean_obs, clean_mean, clean_var = sample_noisy_stats(vec_norm, n_steps=100_000)   # no noise wrapper
    noisy_obs, noisy_mean, noisy_var = sample_noisy_stats(noisy_env, n_steps=100_000)  # with noise

    # Compare stats after running noisy env
    mean_after = vec_norm.obs_rms.mean
    var_after = vec_norm.obs_rms.var

    deltamean = np.abs(mean_after - mean_before)
    deltavar  = np.abs(var_after  - var_before)
    print("\n=== Drift of running statistics ===")
    print(f"|Δmean|  – max: {deltamean.max():.4f}   avg: {deltamean.mean():.4f}")
    print(f"|Δvar |  – max: {deltavar.max():.4f}    avg: {deltavar.mean():.4f}")



    # --- Plotting section ---
    import os
    noise_str = str(NOISE_STD).replace('.', '_')
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    obs_dim = clean_obs.shape[1]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Effect of Additive Noise on Observation Normalization", fontsize=16)

    # Plot running mean and variance before/after noise
    axs[0, 0].plot(mean_before, label="Mean before noise", marker='o')
    axs[0, 0].plot(mean_after, label="Mean after noise", marker='x')
    axs[0, 0].set_title("VecNormalize Running Mean")
    axs[0, 0].legend()
    axs[0, 0].set_xlabel("Observation dimension")
    axs[0, 0].set_ylabel("Mean")
    axs[0, 0].set_ylim(-1.5, 1.5)  # set mean limits

    axs[0, 1].plot(var_before, label="Var before noise", marker='o')
    axs[0, 1].plot(var_after, label="Var after noise", marker='x')
    axs[0, 1].set_title("VecNormalize Running Variance")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel("Observation dimension")
    axs[0, 1].set_ylabel("Variance")
    axs[0, 1].set_ylim(-1.5, 1.5)  # set var limits

    # Plot sampled mean and var (clean vs noisy)
    axs[1, 0].plot(clean_mean, label="Sampled mean (clean)", marker='o')
    axs[1, 0].plot(noisy_mean, label="Sampled mean (noisy)", marker='x')
    axs[1, 0].set_title("Sampled Mean of Observations")
    axs[1, 0].legend()
    axs[1, 0].set_xlabel("Observation dimension")
    axs[1, 0].set_ylabel("Mean")
    axs[1, 0].set_ylim(-1.0, 1.0)  # set mean limits

    axs[1, 1].plot(clean_var, label="Sampled var (clean)", marker='o')
    axs[1, 1].plot(noisy_var, label="Sampled var (noisy)", marker='x')
    axs[1, 1].set_title("Sampled Variance of Observations")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("Observation dimension")
    axs[1, 1].set_ylabel("Variance")
    axs[1, 1].set_ylim(0.0, 3.0)  # set var limits

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot1_path = os.path.join(plots_dir, f"obs_stats_NOISE_STD_{noise_str}.png")
    plt.savefig(plot1_path)
    plt.close(fig)

    # --- Plot histograms for a few dimensions ---
    dims_to_plot = [0, 1, 2]  # Plot first 3 dims for clarity
    fig2 = plt.figure(figsize=(15, 4))
    for i, dim in enumerate(dims_to_plot):
        plt.subplot(1, len(dims_to_plot), i+1)
        plt.hist(clean_obs[:, dim], bins=50, alpha=0.6, label="Clean", density=True)
        plt.hist(noisy_obs[:, dim], bins=50, alpha=0.6, label="Noisy", density=True)
        plt.title(f"Obs dim {dim} histogram")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

    plt.suptitle("Distribution of Normalized Observations (Clean vs Noisy)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot2_path = os.path.join(plots_dir, f"obs_hist_NOISE_STD_{noise_str}.png")
    plt.savefig(plot2_path)
    plt.close(fig2)

    print(f"\nPlots saved to: {plots_dir}")


    # # ------------------------------------------------------------------
    # # 5. Train PPO on the noisy‑normalised obs
    # # ------------------------------------------------------------------
    # model = PPO("MlpPolicy", noisy_env, verbose=1, n_steps=2048, batch_size=64)
    # model.learn(total_timesteps=1_000_000)

    # # ------------------------------------------------------------------
    # # 6. Persist useful artefacts for evaluation/inference
    # # ------------------------------------------------------------------
    # vec_norm.save("ant_vecnorm.pkl")
    # model.save("ppo_ant_noise")


if __name__ == "__main__":
    main()
