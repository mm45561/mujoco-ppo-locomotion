import time

import mujoco
import mujoco.viewer
import torch

from env import BalanceEnv

env = BalanceEnv()
net = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
net.load_state_dict(torch.load("03-rl-pendulum-balance/policy_ppo.pt"))

SLOW = 2.0       # 2.0 = half speed, so the motion is easy to follow
EPISODE = 180    # steps per demo round (3.6 seconds of simulated time)
PUSH_AT = 100    # step at which we push the rod


def restart(sign):
    """Start the rod strongly tilted (0.55 rad is about 31 degrees)."""
    env.reset()
    env.data.qpos[0] = sign * 0.55
    mujoco.mj_forward(env.model, env.data)
    return env._obs()


def policy_action(obs):
    with torch.no_grad():
        return net(torch.from_numpy(obs)).item()


sign = 1
obs = restart(sign)
t = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 1.25]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15
    viewer.sync()
    time.sleep(1.5)  # a moment to see the tilted start
    while viewer.is_running():
        obs, reward, done = env.step(policy_action(obs))
        t += 1
        if t == PUSH_AT:
            env.data.qvel[0] += -2.0 * sign  # a sudden push
        viewer.sync()
        time.sleep(SLOW * env.frame_skip * env.model.opt.timestep)
        if done or t >= EPISODE:
            sign = -sign  # alternate the side
            obs = restart(sign)
            t = 0
            viewer.sync()
            time.sleep(1.5)
