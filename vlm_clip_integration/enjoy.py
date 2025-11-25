import time
import argparse
import gymnasium as gym
import numpy as np

# 1. Import RecurrentPPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from clip_embedder import load_clip
# 2. Import the DYNAMIC wrapper
from dict_obs_wrapper import AddDynamicGoalVecDictObs 

def make_eval_env(env_id, tile_size=8, seed=0, clip_model=None, tokenizer=None, device=None):
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    
    def _init():
        e = gym.make(env_id, render_mode="human", tile_size=tile_size)
        e = RGBImgPartialObsWrapper(e)
        e = ImgObsWrapper(e)
        e.reset(seed=seed)
        
        # 3. Use the DYNAMIC wrapper
        e = AddDynamicGoalVecDictObs(
            e, 
            clip_model=clip_model, 
            clip_tokenizer=tokenizer, 
            clip_device=device
        )
        return e
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/agent_phase1_goto.zip")
    parser.add_argument("--env", default="MiniGrid-ObstructedMaze-2Dlh-v0")
    args = parser.parse_args()

    print("Loading CLIP for goal generation...")
    model, tokenizer, preprocess, device = load_clip("ViT-B-32", "openai")

    print(f"Creating environment: {args.env}")
    env_fn = make_eval_env(args.env, clip_model=model, tokenizer=tokenizer, device=device)
    env = DummyVecEnv([env_fn])
    env = VecTransposeImage(env)

    print(f"Loading RecurrentPPO model from {args.model_path}...")
    # 4. Load with RecurrentPPO
    agent = RecurrentPPO.load(args.model_path, env=env)

    print("Starting simulation... (Press Ctrl+C to stop)")
    
    # 5. Initialize LSTM state (start empty)
    lstm_states = None
    obs = env.reset()
    
    try:
        while True:
            # 6. Pass LSTM states to predict() and get new states back
            action, lstm_states = agent.predict(
                obs, 
                state=lstm_states, 
                deterministic=True
            )
            
            obs, rewards, dones, infos = env.step(action)
            env.render()
            time.sleep(0.1)
            
            if dones[0]:
                print("Episode finished! Resetting...")
                # Note: LSTM state is automatically reset by PPO on 'done'
                # but we reset our 'obs' variable
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()