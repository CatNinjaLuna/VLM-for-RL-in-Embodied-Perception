# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Implements a custom Stable-Baselines3 features extractor that uses 
# spatial softmax to fuse visual and language-goal representations.

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Converts a feature map into a set of spatial coordinates 
    representing the (x, y) centers of the features.
    """
    def __init__(self, height, width, temperature=None):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature or 1.0
        
        # Create coordinate grids
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height)
        )

        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(height * width)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(height * width)).float())

    def forward(self, feature_map):
        N, C, H, W = feature_map.shape
        
        flat = feature_map.view(N, C, -1)
        
        # Softmax over the spatial dimensions
        softmax_attention = F.softmax(flat / self.temperature, dim=2)
        
        # Calculate expected X and Y coordinates
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2)
        
        # Concatenate coordinates
        return torch.cat((expected_x, expected_y), dim=1)

class ImagePlusGoalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out=256, goal_out=64, fused=256):
        super().__init__(observation_space, features_dim=fused)

        img_space = observation_space["image"]
        goal_space = observation_space["goal"]

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        
        with torch.no_grad():
            if img_space.shape[0] == 3:
                dummy = torch.zeros(1, *img_space.shape)
            else:
                h, w, c = img_space.shape
                dummy = torch.zeros(1, c, h, w)
            
            cnn_out_map = self.cnn(dummy)
            n, c, h, w = cnn_out_map.shape
            
        # spatial softmax layer
        self.attention = SpatialSoftmax(h, w)
        vision_feature_dim = c * 2 

        # projection layers
        self.img_head = nn.Sequential(
            nn.Linear(vision_feature_dim, cnn_out), 
            nn.LayerNorm(cnn_out),
            nn.ReLU()
        )
        
        self.goal_head = nn.Sequential(
            nn.Linear(goal_space.shape[0], goal_out), 
            nn.LayerNorm(goal_out),
            nn.ReLU()
        )
        
        self.fuse = nn.Sequential(
            nn.Linear(cnn_out + goal_out, fused), 
            nn.ReLU()
        )

    def forward(self, obs):
        features_map = self.cnn(obs["image"])
        spatial_coords = self.attention(features_map)
        img_lat = self.img_head(spatial_coords)
        goal_lat = self.goal_head(obs["goal"])

        return self.fuse(torch.cat([img_lat, goal_lat], dim=1))