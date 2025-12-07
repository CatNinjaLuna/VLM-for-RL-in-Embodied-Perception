"""
features_extractor.py

Implements a custom Stable-Baselines3 features extractor that uses
Spatial Softmax Attention to fuse visual and language-goal representations.

The module expects a Dict observation:
    "image" -> RGB array (C, H, W)
    "goal"  -> CLIP text-embedding vector (D,)

Architecture:
1. Vision: CNN -> Spatial Softmax -> Coordinate Features (x, y)
   This forces the agent to learn explicit object locations.
2. Language: MLP -> Goal Features
3. Fusion: Concatenation -> MLP
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Converts a feature map of shape (N, C, H, W) into a set of
    spatial coordinates (N, C*2) representing the (x, y) centers
    of the features.
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
        # Register buffers so they are moved to GPU automatically
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(height * width)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(height * width)).float())

    def forward(self, feature_map):
        # feature_map: (N, C, H, W)
        N, C, H, W = feature_map.shape
        
        # Flatten H, W -> (N, C, H*W)
        flat = feature_map.view(N, C, -1)
        
        # Softmax over the spatial dimensions
        # effectively creating a probability map for each channel
        softmax_attention = F.softmax(flat / self.temperature, dim=2)
        
        # Calculate expected X and Y coordinates
        # (N, C, H*W) * (H*W) -> (N, C)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2)
        
        # Concatenate coordinates -> (N, C*2)
        return torch.cat((expected_x, expected_y), dim=1)

import numpy as np # Needed for meshgrid

class ImagePlusGoalExtractor(BaseFeaturesExtractor):
    """
    Expects observation_space: Dict with
      image: Box(C, H, W)
      goal:  Box(D,)
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out=256, goal_out=64, fused=256):
        # Calculate features_dim based on new attention mechanism
        super().__init__(observation_space, features_dim=fused)

        img_space = observation_space["image"]
        goal_space = observation_space["goal"]

        # 1. The CNN Backbone
        # We use a slightly deeper CNN to get better feature maps before attention
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        
        # 2. Compute the shape of the feature map (C, H, W) after CNN
        with torch.no_grad():
            if img_space.shape[0] == 3: # Channel first
                dummy = torch.zeros(1, *img_space.shape)
            else: # Channel last fallback
                h, w, c = img_space.shape
                dummy = torch.zeros(1, c, h, w)
            
            cnn_out_map = self.cnn(dummy)
            n, c, h, w = cnn_out_map.shape
            # print(f"CNN Output Shape: {c} channels, {h}x{w} map")
            
        # 3. Spatial Softmax Layer
        # Input: (N, 64, H, W) -> Output: (N, 64*2) = (N, 128) coordinates
        self.attention = SpatialSoftmax(h, w)
        
        # The output dimension is 2 coords (x,y) per channel
        vision_feature_dim = c * 2 

        # 4. Projection Layers
        self.img_head = nn.Sequential(
            nn.Linear(vision_feature_dim, cnn_out), 
            nn.LayerNorm(cnn_out), # Normalize for stability
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
        # 1. Process Image
        # CNN -> Feature Maps -> Spatial Softmax -> Coordinates
        features_map = self.cnn(obs["image"])
        spatial_coords = self.attention(features_map)
        img_lat = self.img_head(spatial_coords)
        
        # 2. Process Goal
        goal_lat = self.goal_head(obs["goal"])
        
        # 3. Fuse
        return self.fuse(torch.cat([img_lat, goal_lat], dim=1))