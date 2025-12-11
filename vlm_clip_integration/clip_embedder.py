# Author: Peiyao Tao, Carolina Li
# Date: 12/11/2025
# Class: CS 7180 Advanced Perception
# Description: Utility module for loading a pretrained CLIP model and converting 
# goal text into a normalized embedding vector. This embedding serves as the 
# language-conditioned goal representation for the reinforcement learning agent.

import torch
import open_clip

def load_clip(model_name="ViT-B-32", pretrained="openai", device=None):
    """
    Loads the CLIP model, tokenizer, and preprocessing transforms from OpenCLIP.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, tokenizer, preprocess, device

@torch.no_grad()
def text_to_vec(model, tokenizer, device, text: str):
    """ 
    Encodes a text prompt into a feature vector normalized to unit length, 
    suitable for concatenation with visual or policy features in downstream training. 
    """
    toks = tokenizer([text])
    toks = toks.to(device)
    t = model.encode_text(toks)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.squeeze(0).detach().cpu().numpy()
