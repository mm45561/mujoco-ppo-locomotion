# Pendulum balance with REINFORCE

A rod on a hinge learns to balance itself upright, using only a reward signal.
Built with MuJoCo (physics) and PyTorch (the policy network).

## The problem
Keep a rod standing upright on a hinge. Upright is an unstable equilibrium: any small
tilt grows on its own under gravity, so the controller has to keep correcting it.
The motor is limited to ±1 N·m, which is weaker than gravity when the rod is horizontal
(about 1.6 N·m), so it cannot just lift the rod. It has to balance it.

## What is in this folder
| File | What it does |
|---|---|
| `balance.xml` | MuJoCo model: a rod, a hinge with small friction, and a torque-limited motor |
| `env.py` | The environment: `reset()` and `step()`, with observation, reward and fall detection |
| `random_agent.py` | Baseline: random torques |
| `pd_agent.py` | Baseline: a hand-written PD controller (the "answer key") |
| `train_reinforce.py` | Trains a neural-network policy with REINFORCE |
| `play.py` | Loads the trained policy and shows it in the MuJoCo viewer |
| `policy.pt` | The trained policy weights |

**Environment details**
- Observation: `cos(angle)`, `sin(angle)`, angular velocity
- Action: motor torque in [-1, 1]
- Reward per step: `cos(angle) - 0.01 * torque^2`
- An episode ends when the rod tilts more than 0.8 rad, or after 200 steps (4 seconds)

## Results
| Agent | Average steps before falling (max 200) |
|---|---|
| Random torques | 29.7 |
| Hand-written PD controller (kp=8, kd=1) | 200 |
| REINFORCE policy | survives 200 steps after about 40 updates, but rests tilted at about 21 degrees |

During training the average episode length went from 36 at the start to 200 at update 40.
It dipped twice later (194 at update 90, 161 at update 180) before recovering.

## What I learned
- A policy is a small neural network (3 inputs, 32 hidden units, 1 output) that maps
  observations to a torque. During training it samples actions around its mean, which gives exploration.
- REINFORCE raises the probability of actions that were followed by better-than-average
  returns. It uses the same `backward()` and `optimizer.step()` loop as ordinary supervised training.
- The occasional dips come from an update that is too large. That is the problem PPO
  is designed to prevent by limiting how far one update can move the policy.
- The hand-written PD controller shows the task is solvable, and the learned policy reaches
  the same level without being given that formula.

## Run it
```
uv sync
uv run python 03-rl-pendulum-balance/random_agent.py
uv run python 03-rl-pendulum-balance/pd_agent.py
uv run python 03-rl-pendulum-balance/train_reinforce.py
uv run python 03-rl-pendulum-balance/play.py
```

*Built step by step with guidance from Claude, as part of my own learning project.*

## REINFORCE vs PPO
Same task, same environment, same policy size (3 inputs, 32 hidden units, 1 output).
`train_ppo.py` changes the learning rule in three ways:
- a **critic** network estimates the expected future reward, so each action is judged by
  how much better it turned out than expected (the advantage)
- the update is **clipped**: an action's probability can change by about 20% at most per update
- each batch of data is used for **4 learning passes** instead of one

| | REINFORCE | PPO |
|---|---|---|
| First reached 200 steps | update 40 | update 60 |
| After reaching 200 | dipped twice (194 at update 90, 161 at update 180) | stayed at 200 until update 200 |
| Learning rate | 0.01 | 0.003 |

PPO is slower to start but stable once it has learned: it does not suffer the sudden
drops that REINFORCE showed. This is one run of each, with the average printed every
10 updates, so it is a first impression and not a statistical comparison.

Extra files: `train_ppo.py` and `policy_ppo.pt` (the PPO policy weights).

## A correction: surviving is not the same as upright
"200 steps" only means the rod never tilted past 0.8 rad. It does not mean the rod is upright.
I measured where each policy holds the rod over the last 50 steps of 20 episodes
(`measure_angle.py`):

| Policy | Rests at |
|---|---|
| REINFORCE | -0.360 rad (-20.6 degrees) |
| PPO | -0.115 rad (-6.6 degrees) |

The reward `cos(angle)` is almost flat near upright (0.94 at 20 degrees, 1.0 at 0 degrees),
so a policy collects nearly the full reward by holding a tilt. The agent optimizes the reward
I wrote, not what I meant. A natural next experiment is a reward that punishes the angle more
strongly, for example `-angle^2`. For comparison, the hand-written PD controller holds the rod
nearly upright. PPO's resting angle also changes between training runs: on another run of the
same script it rested at about 1.5 degrees.
