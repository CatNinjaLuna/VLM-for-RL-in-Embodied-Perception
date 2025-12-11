# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Main training entry point for PPO with CLIP goal embeddings.

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
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from clip_embedder import load_clip
from dict_obs_wrapper import AddDynamicGoalVecDictObs
from features_extractor import ImagePlusGoalExtractor
from reward_shaping_wrapper import RewardShapingWrapper

def make_rgb_minigrid(env_id: str, tile_size: int = 8, seed: int = 0):
    """ Factory to create RGB MiniGrid environments. """
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    def _init():
        e = gym.make(env_id, render_mode="rgb_array", tile_size=tile_size)
        e = RGBImgPartialObsWrapper(e)
        e = ImgObsWrapper(e)
        e.reset(seed=seed)
        return e
    return _init

def main():
    # Parse command line arguments for flexible training
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-ObstructedMaze-2Dlh-v0")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="./logs/recurrent_clip_vlm")
    parser.add_argument("--model_out", type=str, default="./models/recurrent_ppo_clip_vlm.zip")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a .zip model file to load and continue training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--use_reward_shaping", action="store_true", help="Enable reward shaping wrapper")
    parser.add_argument("--rew_key", type=float, default=0.0, help="Bonus for first key pickup")
    parser.add_argument("--rew_door", type=float, default=0.0, help="Bonus for first door unlock")
    parser.add_argument("--rew_box", type=float, default=0.0, help="Bonus for first box open")
    parser.add_argument("--rew_ball", type=float, default=0.0, help="Bonus for first ball pickup")
    
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Load CLIP
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")

    # Build vectorized envs
    set_random_seed(args.seed)
    def env_fn():
        e = make_rgb_minigrid(args.env, args.tile_size, seed=args.seed)()
        
        if args.use_reward_shaping:
            e = RewardShapingWrapper(
                e, 
                rew_key=args.rew_key,
                rew_door=args.rew_door,
                rew_box=args.rew_box,
                rew_ball=args.rew_ball
            )
        
        e = AddDynamicGoalVecDictObs(e, model, tokenizer, device)
        return e

    # RecurrentPPO requires DummyVecEnv to maintain LSTM order
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecTransposeImage(vec_env)

    print("Creating separate evaluation environment...")
    def eval_env_fn():
        e = make_rgb_minigrid(args.env, args.tile_size, seed=args.seed + 1)()
        e = AddDynamicGoalVecDictObs(e, model, tokenizer, device)
        e = Monitor(e)
        return e

    # Use AddDynamicGoalVecDictObs for the eval env too
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecTransposeImage(eval_env)

    total_timesteps_for_learn = args.steps

    # Initialize or Load Model
    if args.load_model:
        print(f"--- RESUMING TRAINING FROM: {args.load_model} ---")
        model = RecurrentPPO.load(
            args.load_model,
            env=vec_env,
            device="cuda"
        )
        # Configure logger to continue appending to the same logs
        new_logger = utils.configure_logger(model.verbose, args.logdir, "RecurrentPPO", reset_num_timesteps=False)
        model.set_logger(new_logger)

        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = args.lr
        
        model.ent_coef = args.ent_coef
        print(f"Applied new hyperparameters: lr={model.learning_rate}, ent_coef={model.ent_coef}")
        
        total_timesteps_for_learn = model.num_timesteps + args.steps
        print(f"Current steps: {model.num_timesteps}. Adding {args.steps} new steps.")

    else:
        print("--- STARTING NEW TRAINING ---")
        policy_kwargs = dict(
            features_extractor_class=ImagePlusGoalExtractor,
        )
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
            n_steps=2048,
            batch_size=64,
            learning_rate=args.lr, 
            ent_coef=args.ent_coef
        )

    checkpoint_freq = max(50000 // args.n_envs, 1)
    eval_freq = max(25000 // args.n_envs, 1)
    print(f"Checkpointing enabled: Saving backups every {checkpoint_freq * args.n_envs} total steps.")
    print(f"Evaluation enabled: Running every {eval_freq * args.n_envs} total steps.")
    
    if args.use_reward_shaping:
        print("REWARD SHAPING: ENABLED")
    else:
        print("REWARD SHAPING: DISABLED")

    checkpoint_callback = CheckpointCallback(
      save_freq=checkpoint_freq,
      save_path=os.path.join(args.logdir, "checkpoints"),
      name_prefix="rl_model",
      save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
      eval_env,
      best_model_save_path=os.path.join(args.logdir, "best_model"),
      log_path=args.logdir,
      eval_freq=eval_freq,
      deterministic=True,
      render=False
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps_for_learn,
        reset_num_timesteps=False,
        callback=callback,
        progress_bar=True
    )
    
    print(f"[Info] Saving FINAL model to {args.model_out}")
    model.save(args.model_out)

if __name__ == "__main__":
    main()