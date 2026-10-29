import math

import mujoco
import numpy as np


class FurutaEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("04-furuta/furuta.xml")
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10    # one decision every 0.02 s
        self.max_steps = 300    # 6 seconds
        self.fall_angle = 0.8   # the rod has fallen beyond this (rad)
        self.arm_limit = 2.5    # the arm has spun too far beyond this (rad)
        self.steps = 0

    def _obs(self):
        arm, pend = self.data.qpos[0], self.data.qpos[1]
        arm_v, pend_v = self.data.qvel[0], self.data.qvel[1]
        return np.array(
            [math.cos(arm), math.sin(arm), arm_v, math.cos(pend), math.sin(pend), pend_v],
            dtype=np.float32,
        )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = np.random.uniform(-0.2, 0.2)  # rod starts slightly tilted
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.data.ctrl[0] = float(np.clip(action, -1, 1)) * 0.15
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        arm, pend = self.data.qpos[0], self.data.qpos[1]
        arm_v = self.data.qvel[0]
        reward = math.cos(pend) - 0.1 * arm ** 2 - 0.01 * arm_v ** 2
        fell = abs(pend) > self.fall_angle or abs(arm) > self.arm_limit
        done = fell or self.steps >= self.max_steps
        return self._obs(), reward, done
