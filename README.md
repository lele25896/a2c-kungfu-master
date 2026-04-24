# A2C – KungFu Master (Atari)

Advantage Actor-Critic (A2C) agent trained on the Atari game **KungFu Master** using PyTorch and Gymnasium.

## Results

| Version | Reward @ 10k iterations |
|---------|------------------------|
| Course baseline | ~1,380 |
| This implementation | **13,420** |

The agent was also already scoring **7,620** at 3,000 iterations, showing fast and stable convergence.

## Key improvements over the course baseline

### 1. N-step returns (n=5)
Replaced single-step TD(0) targets with 5-step bootstrapped returns:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + γ⁴·r_{t+4} + γ⁵·V(s_{t+5})
```

Each update uses `n_steps × n_envs = 5 × 16 = 80` transitions with significantly lower variance.

### 2. 84×84 input resolution
Standard Atari resolution instead of 42×42, giving the network 4× more spatial information per frame.

### 3. Correct logits flow
Removed erroneous double softmax — the network now returns raw logits, with softmax applied once at the correct point in `act()` and `step()`.

### 4. Gradient clipping
`clip_grad_norm_(parameters, 0.5)` prevents gradient explosions during training.

### 5. Correct episode termination
Uses `terminated or truncated` (Gymnasium API) instead of a single `done` flag.

### 6. Environment: `KungFuMaster-v4`
Uses `v4` (no sticky actions, fixed frameskip=4) matching the deterministic environment of the original course, instead of `v5` which has 25% sticky actions.

## Architecture

```
Input: (4, 84, 84)  — 4 stacked grayscale frames

Conv2d(4  → 32, kernel=3, stride=2)  → ReLU   # (32, 41, 41)
Conv2d(32 → 32, kernel=3, stride=2)  → ReLU   # (32, 20, 20)
Conv2d(32 → 32, kernel=3, stride=2)  → ReLU   # (32,  9,  9)
Flatten → Linear(2592, 128)          → ReLU

Actor head:  Linear(128, n_actions)  → softmax   → action probabilities
Critic head: Linear(128, 1)                      → state value V(s)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (Adam) |
| Discount factor γ | 0.99 |
| N-step returns | 5 |
| Parallel environments | 16 |
| Entropy coefficient | 0.001 |
| Gradient clip norm | 0.5 |
| Reward scaling | ×0.01 |
| Training iterations | 10,000 |

## Installation

```bash
conda create -n torch_env python=3.10
conda activate torch_env
pip install torch torchvision
pip install gymnasium "gymnasium[atari,accept-rom-license]"
pip install opencv-python imageio tqdm
```

## Usage

Open and run `A3C_for_Kung_Fu_Complete_Code.ipynb` cell by cell.  
Training prints average reward every 1,000 iterations.  
After training, `show_video_of_model(agent)` saves a `video.mp4` of the agent playing.

## Dependencies

- Python 3.10+
- PyTorch
- Gymnasium + ALE (Atari Learning Environment)
- OpenCV (`cv2`)
- NumPy, imageio, tqdm
