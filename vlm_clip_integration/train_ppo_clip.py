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

# 1. Import RecurrentPPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
# 2. Import DummyVecEnv (Required for RecurrentPPO)
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from clip_embedder import load_clip
# 3. Import the Dynamic Wrapper
from dict_obs_wrapper import AddDynamicGoalVecDictObs
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
    parser.add_argument("--env", default="MiniGrid-ObstructedMaze-2Dlh-v0")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="./logs/recurrent_clip_vlm")
    parser.add_argument("--model_out", type=str, default="./models/recurrent_ppo_clip_vlm.zip")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a .zip model file to load and continue training.")
    
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # 1) Load CLIP
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")

    # 2) Build vectorized envs
    set_random_seed(args.seed)
    def env_fn():
        e = make_rgb_minigrid(args.env, args.tile_size, seed=args.seed)()
        e = AddDynamicGoalVecDictObs(
            e,
            clip_model=model,
            clip_tokenizer=tokenizer,
            clip_device=device
        )
        return e

    # RecurrentPPO requires DummyVecEnv to maintain LSTM order
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecTransposeImage(vec_env)

    print("Creating separate evaluation environment...")
    eval_env_fn = make_rgb_minigrid(args.env, args.tile_size, seed=args.seed + 1)
    # Use AddDynamicGoalVecDictObs for the eval env too
    eval_env = DummyVecEnv([lambda: AddDynamicGoalVecDictObs(eval_env_fn(), model, tokenizer, device)])
    eval_env = VecTransposeImage(eval_env)

    total_timesteps_for_learn = args.steps

    # 3) Initialize or Load Model
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

        model.learning_rate = 1e-4 
        model.ent_coef = 0.02
        print(f"Applied new hyperparameters: lr={model.learning_rate}, ent_coef={model.ent_coef}")
        total_timesteps_for_learn = model.num_timesteps + args.steps
        print(f"Current steps: {model.num_timesteps}. Adding {args.steps} new steps.")

        # This is the most important part: update the optimizer itself
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = model.learning_rate

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
            
            # Tweak these if you want to force more exploration (Option 1):
            learning_rate=1e-4, 
            ent_coef=0.02
        )

    print(f"Target Total Timesteps: {args.steps}")

    # Callbacks
    checkpoint_freq = max(50000 // args.n_envs, 1)
    eval_freq = max(25000 // args.n_envs, 1)
    print(f"Checkpointing enabled: Saving backups every {checkpoint_freq * args.n_envs} total steps.")
    print(f"Evaluation enabled: Running every {eval_freq * args.n_envs} total steps.")
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

    # 4) Train
    # reset_num_timesteps=False ensures we add to the previous count
    model.learn(
        total_timesteps=total_timesteps_for_learn,
        reset_num_timesteps=False,
        callback=callback,
        progress_bar=True
    )
    
    print(f"[Info] Saving model to {args.model_out}")
    model.save(args.model_out)

if __name__ == "__main__":
    main()