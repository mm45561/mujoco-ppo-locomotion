import time

import mujoco
import mujoco.viewer
import torch

from env import FurutaEnv

env = FurutaEnv()
net = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
net.load_state_dict(torch.load("04-furuta/policy_ppo.pt"))

SLOW = 2.0      # 2.0 = half speed, so the motion is easy to follow
EPISODE = 280   # steps per demo round
PUSH_AT = 150   # step at which we push the rod
TILT = 0.28     # starting tilt of the rod (rad), about 16 degrees


def restart(sign):
    env.reset()
    env.data.qpos[1] = sign * TILT
    mujoco.mj_forward(env.model, env.data)
    return env._obs()


def policy_action(obs):
    with torch.no_grad():
        return net(torch.from_numpy(obs)).item()


sign = 1
obs = restart(sign)
t = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.lookat[:] = [0.1, 0, 0.42]
    viewer.cam.distance = 1.1
    viewer.cam.azimuth = 50
    viewer.cam.elevation = -20
    viewer.sync()
    time.sleep(1.5)  # a moment to see the tilted start
    while viewer.is_running():
        obs, reward, done = env.step(policy_action(obs))
        t += 1
        if t == PUSH_AT:
            env.data.qvel[1] += 1.5 * sign  # a sudden push on the rod
        viewer.sync()
        time.sleep(SLOW * env.frame_skip * env.model.opt.timestep)
        if done or t >= EPISODE:
            sign = -sign  # alternate the side
            obs = restart(sign)
            t = 0
            viewer.sync()
            time.sleep(1.5)
