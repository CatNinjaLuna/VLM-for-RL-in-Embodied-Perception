import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Import the static wrapper that is actually in your file
from clip_embedder import load_clip, text_to_vec
from dict_obs_wrapper import AddGoalVecDictObs 

def make_eval_env(env_id, tile_size=8, seed=0, goal_vec=None):
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    
    def _init():
        # Use render_mode="human" to see the game window
        e = gym.make(env_id, render_mode="human", tile_size=tile_size)
        e = RGBImgPartialObsWrapper(e)
        e = ImgObsWrapper(e)
        e.reset(seed=seed)
        
        # Use the static wrapper you trained with
        e = AddGoalVecDictObs(e, goal_vec=goal_vec)
        return e
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/ppo_clip_vlm.zip")
    parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0") 
    parser.add_argument("--goal_text", default="go to the red goal")
    args = parser.parse_args()

    print(f"Loading CLIP to generate goal vector for: '{args.goal_text}'")
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")
    
    # Generate the vector manually (Static method)
    goal_vec = text_to_vec(model, tokenizer, device, args.goal_text)

    print(f"Creating environment: {args.env}")
    # Pass the fixed vector to the env creator
    env_fn = make_eval_env(args.env, goal_vec=goal_vec)
    
    # Wrap in DummyVecEnv and VecTransposeImage to match training
    env = DummyVecEnv([env_fn])
    env = VecTransposeImage(env)

    print(f"Loading model from {args.model_path}...")
    try:
        agent = PPO.load(args.model_path, env=env)
    except FileNotFoundError:
        print(f"ERROR: Could not find model at {args.model_path}")
        print("Did you rename the zip file or save it somewhere else?")
        return

    print("Starting simulation... (Press Ctrl+C to stop)")
    obs = env.reset()
    
    try:
        while True:
            # deterministic=True usually gives better performance for evaluation
            action, _states = agent.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            env.render()
            
            time.sleep(0.1) # Slow down so you can see it
            
            if dones[0]:
                print("Episode finished! Resetting...")
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()