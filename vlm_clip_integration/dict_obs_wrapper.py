# vlm_clip_integration/dict_obs_wrapper.py
'''
A Gymnasium wrapper that:
Assumes the underlying env outputs RGB frames (H,W,3) (uint8)
Adds a fixed goal vector to every observation (the CLIP text embedding)
'''
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
