# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Defines a Gymnasium ObservationWrapper that augments a standard image-based 
# environment with a fixed CLIP text-embedding vector representing the goal.

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch

class AddDynamicGoalVecDictObs(gym.ObservationWrapper):
    """
    Wraps an env to add a goal vector, but makes the goal dynamic.
    At the start of each episode, this wrapper:
      1. Resets the underlying MiniGrid env.
      2. Reads the env's new mission string.
      3. Uses the CLIP model to compute the embedding for this mission.
      4. Injects this vector into the observation dictionary.
    """
    def __init__(self, env: gym.Env, clip_model, clip_tokenizer, clip_device):
        super().__init__(env)
        
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.clip_device = clip_device

        self.goal_vec_cache = {}
        dummy_vec = self._text_to_vec("placeholder")
        self.current_goal_vec = dummy_vec.astype(np.float32)

        img_space = self.env.observation_space
        if not (isinstance(img_space, spaces.Box) and len(img_space.shape) == 3):
            raise ValueError("Underlying env must have Box image obs with shape (H,W,3)")
            
        self.observation_space = spaces.Dict({
            "image": img_space,
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=dummy_vec.shape, dtype=np.float32)
        })

    @torch.no_grad()
    def _text_to_vec(self, text: str):
        """Encodes a text string into a normalized CLIP vector."""
        if text in self.goal_vec_cache:
            return self.goal_vec_cache[text]
        
        toks = self.clip_tokenizer([text]).to(self.clip_device)
        t = self.clip_model.encode_text(toks)
        t = t / t.norm(dim=-1, keepdim=True)
        vec = t.squeeze(0).detach().cpu().numpy()
        
        self.goal_vec_cache[text] = vec
        return vec

    def reset(self, **kwargs):
        """
        Reset the env and compute the new goal vector
        from the env's `mission` string.
        """
        obs, info = self.env.reset(**kwargs)
        mission_text = self.env.unwrapped.mission
        self.current_goal_vec = self._text_to_vec(mission_text).astype(np.float32)
        
        return {"image": obs, "goal": self.current_goal_vec.copy()}, info

    def observation(self, obs):
        """
        Return the dict observation. The goal vector
        was already updated in reset().
        """
        return {"image": obs, "goal": self.current_goal_vec.copy()}