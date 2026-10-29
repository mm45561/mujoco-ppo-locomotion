# mujoco-ppo-locomotion

PPO reinforcement learning for legged robot locomotion in MuJoCo, built from scratch,
step by step. The goal is to train a walking policy for the Microduck robot with Mjlab.

## What is done

| Stage | Folder | What it shows |
|---|---|---|
| PyTorch basics | [`01-pytorch-basics`](01-pytorch-basics) | Tensors, autograd, gradient descent, and a small network that learns `sin(x)` |
| MuJoCo pendulum | [`02-mujoco-pendulum`](02-mujoco-pendulum) | Writing a model, simulating it, friction, and a position motor (PD control) |
| MuJoCo 2-link leg | [`02-mujoco-2link-leg`](02-mujoco-2link-leg) | Joint chains, sine-wave gait, a free body falling onto a floor |
| RL basics | [`03-rl-basics`](03-rl-basics) | A corridor world, a random baseline (30 steps), and Q-learning (5 steps) |
| Pendulum balance | [`03-rl-pendulum-balance`](03-rl-pendulum-balance) | A custom MuJoCo environment, REINFORCE and PPO policies that balance a rod |

## Result so far
A rod on a hinge with a weak motor learns to balance itself upright from reward alone.

| Agent | Average steps before falling (max 200) |
|---|---|
| Random torques | 29.7 |
| Hand-written PD controller | 200 |
| REINFORCE policy | reaches 200 after about 40 updates, with occasional dips |
| PPO policy | reaches 200 after about 60 updates and stays there |

## Next
- Train the Microduck walking policy with Mjlab (MuJoCo Warp) and PPO
- Read the environment code: observations, rewards, commands, randomisation
- Reward and domain-randomisation experiments, then a final demo

## Setup
Python 3.12 and [uv](https://docs.astral.sh/uv/). Then:
```
uv sync
uv run python 03-rl-pendulum-balance/train_ppo.py
```
Each folder has its own README with the details.

## Credits
The Microduck project and the tutorial that inspired this work:
[Kevin Wood, "Microduck: Getting Started Training a Walking RL PPO Policy with Mjlab"](https://www.youtube.com/watch?v=ZDvCdIFXANw).
This repository is my own learning project, built step by step with guidance from Claude.
