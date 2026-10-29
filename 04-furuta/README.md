# Furuta (rotary) inverted pendulum balanced with PPO

A motor turns an arm around a vertical axis. At the end of the arm, a rod can swing freely
and has no motor of its own. A PPO policy learns to keep the rod upright by moving the arm
underneath it. Built with MuJoCo and PyTorch.

Scope: this folder trains **balancing only** (the rod starts near upright). Swinging the rod
up from hanging down is not done yet.

## What is in this folder
| File | What it does |
|---|---|
| `furuta.xml` | MuJoCo model: a motor-driven arm joint, a free rod joint, and visual-only base and housing |
| `env.py` | The environment: `reset()` and `step()`, with observation, reward and fall detection |
| `random_agent.py` | Baseline: random torques |
| `train_ppo.py` | Trains the policy with PPO (actor and critic networks, GAE, clipped objective) |
| `measure_angle.py` | Measures where the rod really rests, not just how long it survives |
| `demo_ppo.py` | Shows the trained policy in the MuJoCo viewer, with a push |
| `policy_ppo.pt` | The trained policy weights |

## The environment
- Observation (6 numbers): cos and sin of the arm angle, arm speed, cos and sin of the rod angle, rod speed
- Action: motor torque in [-1, 1], scaled to +-0.15 N·m (a weak motor, on purpose)
- Reward per step: `cos(rod angle) - 0.1 * arm_angle^2 - 0.01 * arm_speed^2`
  (the two penalties stop the arm from spinning away)
- An episode ends when the rod tilts more than 0.8 rad, the arm turns more than 2.5 rad,
  or after 300 steps (6 seconds)

## Results (my runs)
| Agent | Result |
|---|---|
| Random torques | falls after 17.6 steps on average (max 300) |
| PPO policy | average episode length reached 300 at update 110 and stayed there until update 300 |

Because surviving is not the same as upright (see the pendulum balance notes), I also measured
the resting pose over 20 episodes with `measure_angle.py`:
- the rod rests at -0.00 degrees from upright
- the arm drifted 2.5 degrees from its start

## Run it
```
uv sync
uv run python 04-furuta/random_agent.py
uv run python 04-furuta/train_ppo.py
uv run python 04-furuta/measure_angle.py
uv run python 04-furuta/demo_ppo.py
```
In the viewer you can double-click the rod, then hold Ctrl and drag with the right mouse
button to push it.

## Inspiration
A real Furuta pendulum trained in simulation and deployed on hardware, from the
Kevin Wood Robotics channel. This folder is only the simulation part.

*Built step by step with guidance from Claude, as part of my own learning project.*
