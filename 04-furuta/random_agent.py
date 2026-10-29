import numpy as np

from env import FurutaEnv

env = FurutaEnv()
lengths = []

for episode in range(100):
    env.reset()
    done = False
    steps = 0
    while not done:
        _, reward, done = env.step(np.random.uniform(-1, 1))  # a random "policy"
        steps += 1
    lengths.append(steps)

print("average steps before falling (random agent):", sum(lengths) / len(lengths))
