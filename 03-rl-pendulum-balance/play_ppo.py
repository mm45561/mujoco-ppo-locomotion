import time

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

obs = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 1.25]  # look at the middle of the rod
    viewer.cam.distance = 2.5            # how far the camera is
    viewer.cam.azimuth = 90              # side view
    viewer.cam.elevation = -15            # level with the rod:
    while viewer.is_running():
        with torch.no_grad():
            action = net(torch.from_numpy(obs)).item()  # the policy's best guess
        obs, reward, done = env.step(action)
        viewer.sync()
        time.sleep(env.frame_skip * env.model.opt.timestep)  # real-time speed
        if done:
            obs = env.reset()  # a new random tilt every 4 seconds
