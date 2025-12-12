# Vision-Language RL Agent for MiniGrid

**Authors:** Peiyao Tao, Carolina Li  
**Date:** 12/11/2025  
**Class:** CS 7180 Advanced Perception  

## Description
This project implements a Vision-Language Action (VLA) agent capable of solving `MiniGrid` environments by following textual goal instructions. 

The system transitions from a vision-only baseline to a vision-language Reinforcement Learning pipeline. It integrates a frozen **CLIP** model to encode mission strings (e.g., "pick up the red ball") into semantic embeddings. These embeddings are fused with visual features (extracted via CNN and Spatial Softmax) to condition the policy of a **Recurrent PPO (Proximal Policy Optimization)** agent.

### Key Features
* **CLIP Integration:** Uses `open_clip` to generate dynamic goal embeddings from environment mission strings.
* **Custom Observation Wrapper:** Wraps MiniGrid to output a dictionary containing both pixel data (`image`) and goal vectors (`goal`).
* **Dual-Stream Architecture:** A custom `ImagePlusGoalExtractor` fuses visual features (CNN + Spatial Softmax) with text features before passing them to the LSTM policy.
* **Reward Shaping:** Optional auxiliary rewards for intermediate sub-goals (e.g., picking up keys, opening doors) to accelerate training.

---

## Operating System & Environment
* **OS Used:** [Windows 10 / 11]
* **Hardware:** A generic GPU is recommended (Code defaults to `cuda` if available) for the CLIP encoder and PPO training.

---

## Installation / Compilation
To run this project, you must install the required Python dependencies.

1. **Prerequisites:**
   * Python 3.8+
   * PyTorch (with CUDA support if using GPU)

2. **Install Dependencies:**
   Run the following command to install the necessary libraries:
   ```bash
   pip install numpy gymnasium minigrid stable-baselines3 sb3-contrib open_clip_torch matplotlib torch tensorboard
   ```

## Project Structure

* **`train_ppo_clip.py`**: The main training entry point that integrates CLIP goal embeddings into the PPO agent, handles environment creation, and manages logging.
* **`enjoy.py`**: The evaluation script used to visualize the agent's performance in real-time, displaying both global and agent-centric views.
* **`features_extractor.py`**: Implements the custom `ImagePlusGoalExtractor` architecture, utilizing a CNN and Spatial Softmax to fuse visual inputs with language goals.
* **`dict_obs_wrapper.py`**: A Gymnasium wrapper that converts the environment's mission text into a fixed CLIP embedding vector and adds it to the observation dictionary.
* **`clip_embedder.py`**: Utility module responsible for loading the pretrained CLIP model and normalizing text-to-vector encodings.
* **`reward_shaping_wrapper.py`**: A wrapper that provides auxiliary dense rewards for sub-goals such as picking up keys, opening doors, or interacting with boxes.

---

## Execution Instructions

### 1. Training the Agent
To train the agent from scratch, run `train_ppo_clip.py`. You can configure hyperparameters via command-line arguments.

**Basic Training:**
```bash
python train_ppo_clip.py
```

**Training with Reward Shaping and Custom Environment:**
To enable reward shaping (bonuses for keys/doors) and specify a harder environment:
```bash
python train_ppo_clip.py \
  --env MiniGrid-ObstructedMaze-2Dlh-v0 \
  --use_reward_shaping \
  --rew_key 0.1 --rew_door 0.1 \
  --steps 3000000 \
  --logdir ./logs/my_experiment
```

**Resuming Training:** To continue training from a saved checkpoint:
```bash
python train_ppo_clip.py --load_model ./models/recurrent_ppo_clip_vlm.zip
```

### 2. Monitoring Training
The training script automatically logs metrics to TensorBoard within the specified log directory. You can monitor the agent's progress, entropy loss, and value loss by pointing TensorBoard to this folder.

**To view training curves:**
```bash
tensorboard --logdir ./logs/
```

### 3. Evaluating / Visualizing the Agent
To watch the trained agent perform in the environment, run the `enjoy.py` script. This script loads the saved model and renders a GUI window showing the "Global View" and the "Agent View" side-by-side.

Key arguments for evaluation include:
* `--model_path`: Path to the trained agent file.
* `--stochastic`: Optional flag to enable probabilistic sampling (default is deterministic).
* `--seed`: Optional integer to fix the environment generation for reproducible testing.

**Run Evaluation:**
```bash
python enjoy.py --model_path ./models/recurrent_ppo_clip_vlm.zip --env MiniGrid-ObstructedMaze-2Dlh-v0
```

---

## Implementation Details

### Architecture
The agent utilizes a **MultiInputLstmPolicy** (Recurrent PPO) to handle the partial observability of the MiniGrid environment. The feature extraction process is handled by the custom `ImagePlusGoalExtractor` class:

1.  **Visual Processing:** The image observation is passed through a 3-layer CNN. The output feature map is processed by a **Spatial Softmax** layer, which converts feature maps into (x, y) spatial coordinates rather than flattening them directly.
2.  **Goal Processing:** The text goal (mission) is encoded into a 512-dimensional vector using a frozen **CLIP (ViT-B-32)** model. This vector is normalized to unit length.
3.  **Fusion:** The spatial visual features and the goal embedding are projected and concatenated before being fed into the LSTM policy.

### Observation Space
The environment is wrapped to provide a Dictionary observation space containing:
* **`"image"`**: A standard RGB image of the grid (typically 56x56 pixels).
* **`"goal"`**: A dynamic float vector representing the semantic meaning of the current mission string.