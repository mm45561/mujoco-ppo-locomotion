import numpy as np
import torch

from env import FurutaEnv

env = FurutaEnv()
net = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
net.load_state_dict(torch.load("04-furuta/policy_ppo.pt"))

rod_angles, arm_angles = [], []
for episode in range(20):
    obs = env.reset()
    done = False
    rod, arm = [], []
    while not done:
        with torch.no_grad():
            action = net(torch.from_numpy(obs)).item()
        obs, reward, done = env.step(action)
        rod.append(env.data.qpos[1])
        arm.append(env.data.qpos[0])
    rod_angles.append(np.mean(rod[-50:]))  # where the rod rests at the end
    arm_angles.append(np.mean(np.abs(arm[-50:])))  # how far the arm drifted

print(f"rod rests at {np.degrees(np.mean(rod_angles)):+.2f} degrees from upright")
print(f"arm drifted {np.degrees(np.mean(arm_angles)):.1f} degrees from its start")
