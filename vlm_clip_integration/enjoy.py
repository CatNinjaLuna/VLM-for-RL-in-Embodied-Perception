# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Evaluation script for the VLA Agent. Loads a trained RecurrentPPO model, 
# generates goal embeddings via CLIP, and visualizes the agent's performance in real-time.

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from clip_embedder import load_clip
from dict_obs_wrapper import AddDynamicGoalVecDictObs
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

ACTION_NAMES = {0: 'Left', 1: 'Right', 2: 'Fwd', 3: 'Pickup', 4: 'Drop', 5: 'Toggle', 6: 'Done'}

def make_eval_env(env_id, tile_size=8, seed=None, clip_model=None, tokenizer=None, device=None):
    """
    Creates and wraps the evaluation environment to match the training setup.
    Includes:
    1. RGBImgPartialObsWrapper: Converts the symbolic grid to pixel observations (56x56).
    2. ImgObsWrapper: Standardizes the image format.
    3. AddDynamicGoalVecDictObs: Injects the CLIP-encoded language goal into the observation dict.
    """ 
    def _init():
        e = gym.make(env_id, render_mode="rgb_array", tile_size=tile_size)
        e = RGBImgPartialObsWrapper(e)
        e = ImgObsWrapper(e)

        # Set specific seed for reproducible evaluation runs
        if seed is not None:
            e.reset(seed=seed)
        else:
            e.reset()
        
        # Custom wrapper: Embeds the text goal into a vector using CLIP
        e = AddDynamicGoalVecDictObs(
            e, 
            clip_model=clip_model, 
            clip_tokenizer=tokenizer, 
            clip_device=device
        )
        return e
    return _init

def main():
    # Parse command line arguments for flexible testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/agent_phase3_ObstructedMaze_1d1h.zip")
    parser.add_argument("--env", default="MiniGrid-ObstructedMaze-1Dlh-v0")
    parser.add_argument("--stochastic", action="store_true", help="Use sampling instead of deterministic mode")
    parser.add_argument("--seed", type=int, default=None, help="Set to None for random mazes") 
    args = parser.parse_args()

    # Load the CLIP model (frozen) to generate goal embeddings for the environment
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")

    # Create the environment with the correct wrappers
    env_fn = make_eval_env(args.env, clip_model=model, tokenizer=tokenizer, device=device, seed=args.seed)
    env = DummyVecEnv([env_fn])
    env = VecTransposeImage(env)

    # Load the trained Recurrent PPO Agent
    agent = RecurrentPPO.load(args.model_path, env=env)

    print("Starting simulation... (Press Ctrl+C to stop)")
    
    # Initialize LSTM states to None at start of episode
    lstm_states = None
    obs = env.reset()

    # Visualization
    plt.ion() 
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    im_global = ax[0].imshow(dummy_img)
    im_agent = ax[1].imshow(dummy_img)
    
    ax[0].set_title("Global View")
    ax[0].axis('off')
    ax[1].set_title("Agent View")
    ax[1].axis('off')
    plt.tight_layout()

    try:
        while True:
            action, lstm_states = agent.predict(
                obs, 
                state=lstm_states, 
                deterministic=not args.stochastic
            )

            # Adjust 'Done' action to 'Fwd' to prevent premature episode termination
            if action[0] == 6: 
                 action[0] = 2 

            obs, rewards, dones, infos = env.step(action)
            
            unwrapped_env = env.envs[0].unwrapped
            global_view = unwrapped_env.get_frame(tile_size=32, highlight=True)
            agent_view = unwrapped_env.get_pov_render(tile_size=32)

            im_global.set_data(global_view)
            im_agent.set_data(agent_view)
            
            plt.draw()
            plt.pause(0.1) # Adjust to control playing speed
            
            if dones[0]:
                final_reward = rewards[0]
                if final_reward > 0:
                    print(f"\n>>> SUCCESS! (Reward: {final_reward:.2f})")
                else:
                    print("\n>>> FAILURE (Timeout)")
                
                print("Resetting environment...\n")
                
                obs = env.reset()
                lstm_states = None
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        plt.close()

if __name__ == "__main__":
    main()