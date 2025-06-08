import os
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from observation_wrappers import AddExtraObsDims

# 🔧 Paths
RUN_DIR = "runs/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_40-extra_std_1.0/run1/seed0"
MODEL_PATH = os.path.join(RUN_DIR, "antbulletenv-v0-x1-seed0_final.zip")
VECNORM_PATH = os.path.join(RUN_DIR, "vecnormalize.pkl")
STACK_PATH = os.path.join(RUN_DIR, "env_stack.json")

def build_env():
    with open(STACK_PATH, "r") as f:
        stack = json.load(f)

    params = stack["parameters"]
    obs_repeat = params["obs_repeat"]
    obs_noise = params["obs_noise"]
    extra_obs_dims = params["extra_obs_dims"]
    extra_obs_noise_std = params["extra_obs_noise_std"]
    env_wrapper_path = params["env_wrapper"]

    # Load custom wrapper
    env_wrapper_cls = None
    if env_wrapper_path:
        mod, cls = env_wrapper_path.rsplit(".", 1)
        import importlib
        env_wrapper_cls = getattr(importlib.import_module(mod), cls)

    def _make_base():
        env = gym.make("AntBulletEnv-v0", exclude_current_positions_from_observation=False)
        env.reset(seed=0)
        env = gym.wrappers.Monitor(env)
        if obs_repeat > 1:
            from observation_wrappers import ObservationRepeater
            env = ObservationRepeater(env, repeat=obs_repeat)
        if env_wrapper_cls:
            env = env_wrapper_cls(env)
        return env

    raw_env = DummyVecEnv([_make_base])
    raw_env = VecNormalize.load(VECNORM_PATH, raw_env)

    if obs_noise > 0:
        from observation_wrappers import AddGaussianNoise
        raw_env = AddGaussianNoise(raw_env, sigma=obs_noise)

    if extra_obs_dims > 0:
        raw_env = AddExtraObsDims(raw_env,
                                  extra_dims=extra_obs_dims,
                                  std=extra_obs_noise_std)

    return raw_env

def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")

    print(f"📦 Loading model: {MODEL_PATH}")
    env = build_env()
    model = PPO.load(MODEL_PATH, env=env)

    # Check if observations are normalized
    if isinstance(env, VecNormalize):
        print("✅ VecNormalize is used (obs are normalized).")
        print(f"   - training mode: {env.training}")
        print(f"   - mean[:5]: {env.obs_rms.mean[:5]}")
        print(f"   - var[:5]:  {env.obs_rms.var[:5]}")
    else:
        print("❌ VecNormalize is NOT used (obs are raw).")

    # Reset and fetch obs
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"🔎 Observation shape: {obs.shape}")
    obs_dim = obs.shape[1] if obs.ndim == 2 else obs.shape[0]
    print(f"   - total dimensions: {obs_dim}")
    print("   - example obs[:10]:", obs[0][:10])

    # Check for extra dims
    ANT_BASE_OBS_DIM = 28
    if obs_dim > ANT_BASE_OBS_DIM:
        extra = obs[0][ANT_BASE_OBS_DIM:]
        extra_std = np.std(extra)
        print(f"⚠️ Detected {obs_dim - ANT_BASE_OBS_DIM} extra dims.")
        print(f"   - extra dim stats: mean={np.mean(extra):.4f}, std={extra_std:.4f}")
        if extra_std < 2.0:
            print("   ✅ Extra dims appear to be normalized.")
        else:
            print("   ⚠️ Extra dims likely *not* normalized.")
    else:
        print("✅ No extra noise dimensions detected.")

if __name__ == "__main__":
    main()




# import os
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecNormalize
# import gymnasium as gym
# from observation_wrappers import AddExtraObsDims

# # 🔧 Set the full path to your saved model here
# MODEL_PATH = "runs/AntBulletEnv-v0-x1-obs_noise_0.0-extra_dims_40-extra_std_1.0/run1/seed0/antbulletenv-v0-x1-seed0_final.zip"  # ← CHANGE THIS PATH

# def main(model_path):
#     if not os.path.isfile(model_path):
#         raise FileNotFoundError(f"Model file does not exist: {model_path}")

#     print(f"📦 Loading model: {model_path}")
#     model = PPO.load(model_path)

#     env = model.get_env()

#     # Check if observations are normalized
#     if isinstance(env, VecNormalize):
#         print("✅ VecNormalize is used (obs are normalized).")
#         print(f"   - training mode: {env.training}")
#         print(f"   - mean[:5]: {env.obs_rms.mean[:5]}")
#         print(f"   - var[:5]:  {env.obs_rms.var[:5]}")
#     else:
#         print("❌ VecNormalize is NOT used (obs are raw).")

#     # Reset and fetch obs
#     obs = env.reset()
#     if isinstance(obs, tuple):
#         obs = obs[0]

#     print(f"🔎 Observation shape: {obs.shape}")
#     obs_dim = obs.shape[1] if obs.ndim == 2 else obs.shape[0]
#     print(f"   - total dimensions: {obs_dim}")
#     print("   - example obs[:10]:", obs[0][:10])

#     # Check for extra dims (> default Ant obs dim = 28)
#     ANT_BASE_OBS_DIM = 28
#     if obs_dim > ANT_BASE_OBS_DIM:
#         extra = obs[0][ANT_BASE_OBS_DIM:]
#         extra_std = np.std(extra)
#         print(f"⚠️ Detected {obs_dim - ANT_BASE_OBS_DIM} extra dims.")
#         print(f"   - extra dim stats: mean={np.mean(extra):.4f}, std={extra_std:.4f}")
#         if extra_std < 2.0:
#             print("   ✅ Extra dims appear to be normalized.")
#         else:
#             print("   ⚠️ Extra dims likely *not* normalized.")
#     else:
#         print("✅ No extra noise dimensions detected.")

# if __name__ == "__main__":
#     main(MODEL_PATH)
