import numpy as np
import torch

from env import BalanceEnv

env = BalanceEnv()

for name in ["policy.pt", "policy_ppo.pt"]:
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )
    net.load_state_dict(torch.load("03-rl-pendulum-balance/" + name))

    final_angles = []
    for episode in range(20):
        obs = env.reset()
        done = False
        angles = []
        while not done:
            with torch.no_grad():
                action = net(torch.from_numpy(obs)).item()
            obs, reward, done = env.step(action)
            angles.append(env.data.qpos[0])
        final_angles.append(np.mean(angles[-50:]))  # where it rests at the end

    print(f"{name}: rests at {np.mean(final_angles):+.3f} rad ({np.degrees(np.mean(final_angles)):+.1f} degrees)")
