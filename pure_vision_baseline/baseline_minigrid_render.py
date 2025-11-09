#!/usr/bin/env python3
"""
baseline_minigrid_render.py

Week 1 helper script:
- Sets up MiniGrid to produce 64x64 RGB observations
- Trains a PPO baseline for a short run (~10k steps) with CnnPolicy
- Saves sample rendered frames, model checkpoint, and TensorBoard logs

Usage:
    python baseline_minigrid_render.py --env MiniGrid-Empty-8x8-v0 --steps 10000 --n_envs 4 --logdir ./logs/baseline

Requirements (install before running):
    pip install gymnasium minigrid stable-baselines3==2.3.0 tensorboard matplotlib
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def make_minigrid_env(env_id: str, seed: int = 0, tile_size: int = 8):
    """
    Create a MiniGrid env that returns 64x64 RGB observations.
    tile_size=8 on an 8x8 grid => 64x64 image.
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array", tile_size=tile_size)
        # Wrap to expose RGB image in the observation and then keep only the image
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(env_id: str, n_envs: int = 4, seed: int = 0, tile_size: int = 8):
    """
    Vectorized environment with image transposition for CNN policies.
    """
    set_random_seed(seed)
    if n_envs > 1:
        env = SubprocVecEnv([make_minigrid_env(env_id, seed + i, tile_size) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_minigrid_env(env_id, seed, tile_size)])
    # SB3 CNN expects channel-first (C,H,W)
    env = VecTransposeImage(env)
    return env


def save_sample_frames(single_env_id: str, out_dir: str = "./samples", n_frames: int = 8, tile_size: int = 8, seed: int = 0):
    """
    Roll out random actions and save a few rendered frames for sanity check.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make(single_env_id, render_mode="rgb_array", tile_size=tile_size)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    obs, info = env.reset(seed=seed)

    for i in range(n_frames):
        # Render a frame from the env (not the vectorized wrapper)
        frame = env.render()  # returns HxWx3 RGB array
        if frame is not None:
            plt.imsave(os.path.join(out_dir, f"frame_{i:03d}.png"), frame)
        # Take a random step to vary the view
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def evaluate_policy_rollouts(env_id: str, model_path: str, n_episodes: int = 10, tile_size: int = 8, seed: int = 123):
    """
    Quick evaluation loop (greedy actions). Returns average episode length and reward.
    Note: MiniGrid tasks differ; for Empty-8x8, success ~ reaching goal before timeout.
    """
    env = gym.make(env_id, render_mode="rgb_array", tile_size=tile_size)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    # Wrap for channel-first shape for model prediction
    # We'll avoid vector wrapper here; just handle transpose manually for prediction.
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: F401
    model = PPO.load(model_path)

    lengths, rewards, successes = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_len = 0
        ep_rew = 0.0
        while not done:
            # Model expects (N,C,H,W); convert obs (H,W,C)->(C,H,W) and add batch dim
            obs_chw = np.transpose(obs, (2, 0, 1))
            action, _ = model.predict(obs_chw, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_len += 1
            ep_rew += reward
        lengths.append(ep_len)
        rewards.append(ep_rew)
        successes.append(1 if info.get("success", terminated) else 0)

    env.close()
    return {
        "avg_episode_length": float(np.mean(lengths)),
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="./logs/baseline")
    parser.add_argument("--model_out", type=str, default="./models/ppo_minigrid_baseline")
    parser.add_argument("--save_frames", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Optional sanity-check frames (64x64 RGB)
    if args.save_frames:
        print("[Info] Saving sample frames...")
        save_sample_frames(args.env, out_dir="./samples", n_frames=8, tile_size=args.tile_size, seed=args.seed)

    # Build vectorized env
    print("[Info] Building vectorized env...")
    vec_env = build_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, tile_size=args.tile_size)

    # Train PPO
    print(f"[Info] Starting PPO training for {args.steps} steps...")
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.steps)
    model.save(args.model_out)
    print(f"[Info] Saved model to: {args.model_out}")

    # Quick evaluation
    print("[Info] Evaluating policy...")
    metrics = evaluate_policy_rollouts(args.env, args.model_out, n_episodes=5, tile_size=args.tile_size, seed=args.seed)
    print("[Eval] ", metrics)

    print("\nDone. To view TensorBoard logs, run:\n  tensorboard --logdir {} --port 6006".format(args.logdir))


if __name__ == "__main__":
    main()
