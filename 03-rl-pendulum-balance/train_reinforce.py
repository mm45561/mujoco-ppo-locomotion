import numpy as np
import torch

from env import BalanceEnv

torch.manual_seed(0)
np.random.seed(0)

env = BalanceEnv()

# The policy: observation (3 numbers) -> mean torque. The spread (std) is also learned.
net = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
log_std = torch.nn.Parameter(torch.tensor([-0.5]))
optimizer = torch.optim.Adam(list(net.parameters()) + [log_std], lr=0.01)
gamma = 0.97


def run_episode():
    """Play one episode, sampling actions from the policy."""
    obs = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        mean = net(torch.from_numpy(obs))
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum())
        obs, reward, done = env.step(action.item())
        rewards.append(reward)
    return log_probs, rewards


def discounted_returns(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


for iteration in range(201):
    all_log_probs, all_returns, lengths = [], [], []
    for _ in range(10):  # 10 episodes per update
        log_probs, rewards = run_episode()
        all_log_probs += log_probs
        all_returns += discounted_returns(rewards)
        lengths.append(len(rewards))

    returns = torch.tensor(all_returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # who did better than average?
    loss = -(torch.stack(all_log_probs) * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % 10 == 0:
        print(f"iteration {iteration:3d}  average episode length = {np.mean(lengths):6.1f}")

torch.save(net.state_dict(), "03-rl-pendulum-balance/policy.pt")
