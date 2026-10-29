import numpy as np
import torch

from env import FurutaEnv

torch.manual_seed(0)
np.random.seed(0)

env = FurutaEnv()

# Actor: observation -> mean torque (the policy). Critic: observation -> expected future reward.
actor = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
critic = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
log_std = torch.nn.Parameter(torch.tensor([-0.5]))
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=0.003
)

gamma, lam = 0.97, 0.95  # discount, and how far GAE looks ahead
clip = 0.2               # a single update may change an action's probability by about +-20%
epochs = 4               # learn from each batch of data 4 times


def collect(num_episodes):
    """Play episodes with the current policy and record everything."""
    obs_l, act_l, logp_l, adv_l, ret_l, lengths = [], [], [], [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_obs, ep_act, ep_logp, ep_rew = [], [], [], []
        done = False
        while not done:
            with torch.no_grad():
                o = torch.from_numpy(obs)
                dist = torch.distributions.Normal(actor(o), log_std.exp())
                action = dist.sample()
                logp = dist.log_prob(action).sum()
            ep_obs.append(obs)
            ep_act.append(action)
            ep_logp.append(logp)
            obs, reward, done = env.step(action.item())
            ep_rew.append(reward)
        lengths.append(len(ep_rew))

        # If the episode only ended because time ran out, the future still has value.
        still_up = abs(env.data.qpos[1]) <= env.fall_angle and abs(env.data.qpos[0]) <= env.arm_limit
        timed_out = env.steps >= env.max_steps and still_up
        with torch.no_grad():
            values = critic(torch.from_numpy(np.array(ep_obs))).squeeze(1).numpy()
            last_value = critic(torch.from_numpy(obs)).item() if timed_out else 0.0
        values = np.append(values, last_value)

        # GAE: advantage = how much better this step turned out than the critic expected.
        adv = np.zeros(len(ep_rew), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(ep_rew))):
            delta = ep_rew[t] + gamma * values[t + 1] - values[t]
            running = delta + gamma * lam * running
            adv[t] = running
        obs_l += ep_obs
        act_l += ep_act
        logp_l += ep_logp
        adv_l += list(adv)
        ret_l += list(adv + values[:-1])
    return (
        torch.from_numpy(np.array(obs_l)),
        torch.stack(act_l),
        torch.stack(logp_l),
        torch.tensor(adv_l, dtype=torch.float32),
        torch.tensor(ret_l, dtype=torch.float32),
        lengths,
    )


for iteration in range(301):
    obs, actions, old_logp, adv, returns, lengths = collect(10)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(epochs):
        dist = torch.distributions.Normal(actor(obs), log_std.exp())
        new_logp = dist.log_prob(actions).sum(dim=1)
        ratio = (new_logp - old_logp).exp()  # new probability / old probability
        # The PPO trick: never reward moving the ratio outside [1 - clip, 1 + clip].
        policy_loss = -torch.min(
            ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        ).mean()
        value_loss = ((critic(obs).squeeze(1) - returns) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if iteration % 10 == 0:
        print(f"iteration {iteration:3d}  average episode length = {np.mean(lengths):6.1f}")

torch.save(actor.state_dict(), "04-furuta/policy_ppo.pt")
