import numpy as np

from env import BalanceEnv

env = BalanceEnv()
lengths = []

for episode in range(200):
    env.reset()
    done = False
    steps = 0
    while not done:
        action = np.random.uniform(-1, 1)  # a random "policy"
        _, reward, done = env.step(action)
        steps += 1
    lengths.append(steps)

print("average steps before falling (random agent):", sum(lengths) / len(lengths))
print("longest episode:", max(lengths), "of", env.max_steps)
