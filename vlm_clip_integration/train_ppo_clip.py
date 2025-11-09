"""
train_ppo_clip.py

Main training entry point for integrating CLIP goal embeddings into a PPO
agent.  The script:
  1. Loads a pretrained CLIP model and converts a goal text string into a
     fixed embedding vector.
  2. Wraps an RGB MiniGrid environment with AddGoalVecDictObs so that each
     observation contains both image and goal information.
  3. Configures a Stable-Baselines3 PPO model using MultiInputPolicy and the
     custom ImagePlusGoalExtractor to fuse the two modalities.
  4. Trains the agent, logs progress to TensorBoard, and saves model weights.

This script represents the transition from a vision-only baseline to a
vision-language RL pipeline, enabling semantic goal conditioning and serving
as the foundation for future LoRA fine-tuning or BLIP-2 extensions.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

from clip_embedder import load_clip, text_to_vec
from dict_obs_wrapper import AddGoalVecDictObs
from features_extractor import ImagePlusGoalExtractor

def make_rgb_minigrid(env_id: str, tile_size: int = 8, seed: int = 0):
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    def _init():
        e = gym.make(env_id, render_mode="rgb_array", tile_size=tile_size)
        e = RGBImgPartialObsWrapper(e)
        e = ImgObsWrapper(e)  # returns (H,W,3) uint8
        e.reset(seed=seed)
        return e
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal_text", type=str, default="go to the red goal")
    parser.add_argument("--logdir", type=str, default="./logs/clip_vlm")
    parser.add_argument("--model_out", type=str, default="./models/ppo_clip_vlm")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # 1) Load CLIP and turn goal text into a unit vector
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")
    goal_vec = text_to_vec(model, tokenizer, device, args.goal_text)  # (D,)
    goal_dim = goal_vec.shape[0]

    # 2) Build vectorized envs -> wrap with Dict obs (image + goal)
    set_random_seed(args.seed)
    def env_fn():
        e = make_rgb_minigrid(args.env, args.tile_size, seed=args.seed)()
        e = AddGoalVecDictObs(e, goal_vec=goal_vec)
        return e

    vec_env = make_vec_env(env_fn, n_envs=args.n_envs)
    # Important: transpose only the image channel for SB3 CNN
    # SB3's VecTransposeImage expects observation to be image array, but we have Dict.
    # We'll instead use Dict obs directly: SB3 supports Dict for policies since v2 with custom extractors.
    # We must NOT wrap the whole Dict with VecTransposeImage; instead, tell SB3 to auto-transpose in policy.
    # Solution: add a custom policy class or specify `normalize_images=True` and handle transpose in extractor.
    #
    # Simpler approach: convert image from (H,W,3) to (C,H,W) in a wrapper. We'll do it here:

    from stable_baselines3.common.vec_env import VecTranspose
    # Build a transpose wrapper for the "image" key inside Dict observations
    vec_env = VecTranspose(vec_env, op=lambda obs: {"image": np.transpose(obs["image"], (0,3,1,2)),
                                                    "goal": obs["goal"]})

    # 3) PPO + custom features extractor
    policy_kwargs = dict(
        features_extractor_class=ImagePlusGoalExtractor,
        # optional: tweak net sizes via features_extractor_kwargs
        # features_extractor_kwargs=dict(cnn_out=256, goal_out=64, fused=256),
    )

    model = PPO(
        policy="MultiInputPolicy",   # Dict observation
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01
    )

    model.learn(total_timesteps=args.steps)
    model.save(args.model_out)
    print(f"[Info] Saved to {args.model_out}")

if __name__ == "__main__":
    main()
