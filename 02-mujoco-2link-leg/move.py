import math
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("02-mujoco-2link-leg/leg.xml")
data = mujoco.MjData(model)

omega = 2 * math.pi * 0.5  # 0.5 swings per second

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 0.5 * math.sin(omega * data.time)                  # hip target
        data.ctrl[1] = 0.6 * math.sin(omega * data.time - math.pi / 2)    # knee target, a quarter-cycle later
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
