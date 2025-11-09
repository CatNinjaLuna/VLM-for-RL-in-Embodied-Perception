# vlm_clip_integration/dict_obs_wrapper.py
"""
dict_obs_wrapper.py

Defines a Gymnasium ObservationWrapper that augments a standard image-based
environment with a fixed CLIP text-embedding vector representing the goal.
The wrapper converts each observation into a dictionary of the form:
    {"image": (H,W,3) uint8 frame, "goal": (D,) float32 vector}
where D is the embedding dimension (e.g., 512).

This structure allows Stable-Baselines3's MultiInputPolicy to process both
visual and language features within a single PPO pipeline.  The goal vector is
typically constant for an episode and provides high-level semantic context
(e.g., "go to the red cube") while the CNN continues to learn pixel-level
control signals from the image stream.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AddGoalVecDictObs(gym.ObservationWrapper):
    """
    Wrap image observations into a Dict:
      {"image": (H,W,3) uint8, "goal": (D,) float32}
    The 'goal' vector is a fixed CLIP text embedding provided at init.
    """
    def __init__(self, env: gym.Env, goal_vec: np.ndarray):
        super().__init__(env)
        assert isinstance(goal_vec, np.ndarray) and goal_vec.ndim == 1
        self.goal_vec = goal_vec.astype(np.float32)
        # Build dict space
        img_space = self.env.observation_space
        if isinstance(img_space, spaces.Box) and img_space.shape and len(img_space.shape) == 3:
            pass
        else:
            raise ValueError("Underlying env must have Box image obs with shape (H,W,3)")
        self.observation_space = spaces.Dict({
            "image": img_space,  # (H,W,3) uint8
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=self.goal_vec.shape, dtype=np.float32)
        })

    def observation(self, obs):
        return {"image": obs, "goal": self.goal_vec.copy()}
