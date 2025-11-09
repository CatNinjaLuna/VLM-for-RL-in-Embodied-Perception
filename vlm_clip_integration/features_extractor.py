# vlm_clip_integration/features_extractor.py
'''Custom SB3 features extractor that CNN-encodes the image and MLP-encodes the goal vector, then concatenates them.'''
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ImagePlusGoalExtractor(BaseFeaturesExtractor):
    """
    Expects observation_space: Dict with
      image: Box(H,W,3)  -> we internally transpose to (C,H,W) via VecTransposeImage later
      goal:  Box(D,)
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out=256, goal_out=64, fused=256):
        super().__init__(observation_space, features_dim=fused)

        img_space = observation_space["image"]
        goal_space = observation_space["goal"]

        # Simple CNN encoder
        # Note: SB3 will give (N,C,H,W) after VecTransposeImage; first conv expects C=3
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # infer flattened dim
        with th.no_grad():
            h, w, c = img_space.shape
            dummy = th.zeros(1, 3, h, w)  # channel-first
            cnn_dim = self.cnn(dummy).shape[1]

        self.img_head = nn.Sequential(nn.Linear(cnn_dim, cnn_out), nn.ReLU())
        self.goal_head = nn.Sequential(
            nn.Linear(goal_space.shape[0], goal_out), nn.ReLU()
        )
        self.fuse = nn.Sequential(nn.Linear(cnn_out + goal_out, fused), nn.ReLU())

    def forward(self, obs):
        # obs["image"]: (N,C,H,W)
        # obs["goal"]:  (N,D)
        img_lat = self.img_head(self.cnn(obs["image"]))
        goal_lat = self.goal_head(obs["goal"])
        return self.fuse(th.cat([img_lat, goal_lat], dim=1))
