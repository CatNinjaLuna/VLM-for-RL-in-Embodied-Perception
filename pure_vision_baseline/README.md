# PPO Visual Baseline — MiniGrid 64×64 RGB

This experiment establishes a **pure-vision reinforcement learning (RL) baseline** for the _"Fine-Tuning Vision–Language Models for Reinforcement Learning in Embodied Perception Tasks"_ project.

It trains a **PPO agent** on the MiniGrid environment using only 64×64 RGB observations to provide a reference for later Vision–Language Model (VLM) integration.

---

## Environment Overview

-  **Environment:** `MiniGrid-Empty-8x8-v0`
-  **Observation:** 64×64 RGB frames (via `RGBImgPartialObsWrapper` + `ImgObsWrapper`)
-  **Action space:** Discrete (turn left/right, move forward)
-  **Reward:** +1 for reaching goal, 0 otherwise
-  **Training framework:** Stable-Baselines3 (PPO, CNN policy)
-  **Hardware:** CPU-only (Windows)

---

## Setup (Windows PowerShell)

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate

# Install dependencies
pip install gymnasium minigrid stable-baselines3==2.3.0 tensorboard matplotlib

# Run baseline training (10k steps quick test)
python baseline_minigrid_render.py --steps 10000 --n_envs 4 --save_frames

# View training logs
tensorboard --logdir ./logs/baseline --port 6006
```

---

## Output Structure

```
project_root/
│
├── baseline_minigrid_render.py       # main training script
├── models/
│   └── ppo_minigrid_baseline.zip     # saved PPO weights
├── samples/
│   └── frame_000.png ...             # rendered RGB frames
├── logs/
│   └── baseline/                     # TensorBoard logs
└── README.md
```

---

## Training Metrics Summary (10k-Step Run)

| Metric               | Meaning                                           | Value / Interpretation                  |
| -------------------- | ------------------------------------------------- | --------------------------------------- |
| `fps`                | Frames processed per second                       | ~861 fps — efficient CPU training       |
| `approx_kl`          | KL divergence between new/old policies            | 0.0107 → stable updates                 |
| `entropy_loss`       | Policy randomness                                 | –1.94 → still exploring                 |
| `explained_variance` | Correlation between value predictions and returns | –0.95 → value network not learning yet  |
| `value_loss`         | MSE of value estimates                            | 0.0028 → low only because rewards are 0 |
| `avg_episode_length` | Steps before episode ends                         | 256 (max) → agent wanders without goal  |
| `avg_reward`         | Mean episode reward                               | 0.0 → no goal reached                   |
| `success_rate`       | Episodes completed successfully                   | 0.0 → expected early stage              |

---

## Interpretation

-  The model runs correctly but has not yet learned meaningful policy behavior.
-  10k steps are insufficient for PPO convergence on pixel input.
-  Negative `explained_variance` indicates the value network predicts noise (no reward signal yet).
-  Stable `approx_kl` and `entropy_loss` show updates are healthy and exploration is active.
-  This serves as a baseline checkpoint for Week 1.

---

## Next Steps (Week 2 Plan)

1. **Longer training:** increase steps to 200k–500k for meaningful learning.

```bash
python baseline_minigrid_render.py --steps 500000 --n_envs 4
```

2. **Simpler environment:** try `MiniGrid-Empty-5x5-v0` for faster convergence.
3. **Analyze TensorBoard:** track reward and value loss curves.
4. **Prepare for VLM integration:** confirm 64×64 frame extraction works for CLIP/BLIP embedding inputs.

---

## Expected Week 2 Deliverables

-  Trained PPO baseline with non-zero success rate.
-  Reward curves and success-rate plots over training time.
-  Written baseline analysis summary (this document).
-  Saved model checkpoints for comparison with VLM-augmented variants.
