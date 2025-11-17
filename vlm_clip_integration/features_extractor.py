"""
features_extractor.py

Implements a custom Stable-Baselines3 features extractor that fuses visual
and language-goal representations. The module expects a Dict observation
containing:
    "image" -> RGB array processed through a lightweight CNN encoder
    "goal"  -> CLIP text-embedding vector processed by a small MLP

Both feature branches are projected into latent spaces (cnn_out and goal_out)
and concatenated before a final fusion layer produces a shared representation
(features_dim) for the policy and value networks.  This design lets PPO learn
jointly from visual context and semantic goals, forming the bridge between
language understanding (via CLIP) and embodied decision making.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ImagePlusGoalExtractor(BaseFeaturesExtractor):
    """
    Expects observation_space: Dict with
      image: Box(C, H, W)  -> Transposed by VecTransposeImage
      goal:  Box(D,)
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out=256, goal_out=64, fused=256):
        super().__init__(observation_space, features_dim=fused)

        img_space = observation_space["image"]
        goal_space = observation_space["goal"]

        # Simple CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute flattened dim
        with th.no_grad():
            # FIX: Robustly handle channel-first vs channel-last shapes.
            # VecTransposeImage converts to (C, H, W), so shape is likely (3, H, W).
            if img_space.shape[0] == 3:
                # Already (C, H, W) - Correct for PyTorch
                dummy = th.zeros(1, *img_space.shape)
            else:
                # (H, W, C) - Fallback if wrapper wasn't used
                h, w, c = img_space.shape
                dummy = th.zeros(1, c, h, w)

            cnn_dim = self.cnn(dummy).shape[1]

        self.img_head = nn.Sequential(nn.Linear(cnn_dim, cnn_out), nn.ReLU())
        self.goal_head = nn.Sequential(
            nn.Linear(goal_space.shape[0], goal_out), nn.ReLU()
        )
        self.fuse = nn.Sequential(nn.Linear(cnn_out + goal_out, fused), nn.ReLU())

    def forward(self, obs):
        # obs["image"] is already (N, C, H, W) thanks to VecTransposeImage
        img_lat = self.img_head(self.cnn(obs["image"]))
        goal_lat = self.goal_head(obs["goal"])
        return self.fuse(th.cat([img_lat, goal_lat], dim=1))