# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Reward shaping wrapper for MiniGrid environments.

import gymnasium as gym
from minigrid.core.constants import DIR_TO_VEC

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, 
                 rew_key=0.0, 
                 rew_door=0.0, 
                 rew_box=0.0, 
                 rew_ball=0.0):
        super().__init__(env)

        self.rew_key = rew_key
        self.rew_door = rew_door
        self.rew_box = rew_box
        self.rew_ball = rew_ball
        
    def reset(self, **kwargs):
        """ Reset the environment and one-time flags."""
        self.has_picked_key = False
        self.has_picked_ball = False
        self.has_opened_door = False
        self.has_opened_box = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        
        carrying = unwrapped.carrying
        if carrying is not None:
            # Reward for picking the key
            if carrying.type == 'key' and not self.has_picked_key:
                reward += self.rew_key
                self.has_picked_key = True
            
            # Reward for interacting with obstacles
            if carrying.type == 'ball' and not self.has_picked_ball:
                reward += self.rew_ball
                self.has_picked_ball = True

        TOGGLE_IDX = 5 
        if action == TOGGLE_IDX:
            fwd_pos = unwrapped.front_pos
            fwd_cell = unwrapped.grid.get(*fwd_pos)

            # If it's a door and it's now open
            if fwd_cell is not None and fwd_cell.type == 'door' and fwd_cell.is_open:
                if not self.has_opened_door:
                    reward += self.rew_door
                    self.has_opened_door = True
            
            # If it's a box and it's now open
            if fwd_cell is not None and fwd_cell.type == 'box' and fwd_cell.is_open:
                if not self.has_opened_box:
                    reward += self.rew_box
                    self.has_opened_box = True

        return obs, reward, terminated, truncated, info