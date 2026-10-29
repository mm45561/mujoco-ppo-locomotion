import math

from env import BalanceEnv

KP, KD = 8.0, 1.0
env = BalanceEnv()
lengths, returns = [], []

for episode in range(200):
    obs = env.reset()
    done, steps, total = False, 0, 0.0
    while not done:
        theta = math.atan2(obs[1], obs[0])  # recover the angle from sin and cos
        omega = obs[2]
        action = -(KP * theta + KD * omega)  # P term + D term
        obs, reward, done = env.step(action)
        steps += 1
        total += reward
    lengths.append(steps)
    returns.append(total)

print("avg steps:", sum(lengths) / len(lengths), "avg return:", sum(returns) / len(returns))
