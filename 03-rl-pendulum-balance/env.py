import math

import mujoco
import numpy as np


class BalanceEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("03-rl-pendulum-balance/balance.xml")
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10   # 10 physics steps (0.002 s each) per action = 0.02 s
        self.max_steps = 200   # one episode lasts at most 200 actions = 4 seconds
        self.fall_angle = 0.8  # radians: beyond this the pole has fallen
        self.steps = 0

    def _obs(self):
        theta = self.data.qpos[0]
        omega = self.data.qvel[0]
        return np.array([math.cos(theta), math.sin(theta), omega], dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.random.uniform(-0.2, 0.2)  # start slightly tilted
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.data.ctrl[0] = float(np.clip(action, -1, 1))
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        theta = self.data.qpos[0]
        reward = math.cos(theta) - 0.01 * float(self.data.ctrl[0]) ** 2
        fell = abs(theta) > self.fall_angle
        done = fell or self.steps >= self.max_steps
        return self._obs(), reward, done
