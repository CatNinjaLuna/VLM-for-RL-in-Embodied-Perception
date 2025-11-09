"""
clip_embedder.py

Utility module for loading a pretrained CLIP model and converting goal text into
a normalized embedding vector. This embedding serves as the language-conditioned
goal representation for the reinforcement learning agent.

The file provides two key functions:
- load_clip(): loads the CLIP model, tokenizer, and preprocessing transforms
  from OpenCLIP (e.g., ViT-B/32 backbone pretrained on 400M image–text pairs).
- text_to_vec(): encodes a text prompt (e.g., "go to the red cube") into a
  feature vector normalized to unit length, suitable for concatenation with
  visual or policy features in downstream training.

This module isolates CLIP setup and tokenization logic so that the RL training
pipeline (e.g., PPO) can remain lightweight and only operate on fixed-size
numerical goal vectors instead of raw text.
"""

import torch
import open_clip

def load_clip(model_name="ViT-B-32", pretrained="openai", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, tokenizer, preprocess, device

@torch.no_grad()
def text_to_vec(model, tokenizer, device, text: str):
    toks = tokenizer([text])
    toks = toks.to(device)
    t = model.encode_text(toks)  # (1, D)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.squeeze(0).detach().cpu().numpy()  # (D,)
