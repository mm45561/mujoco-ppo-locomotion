import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("02-mujoco-pendulum/pendulum.xml")
data = mujoco.MjData(model)

data.qpos[0] = 1.0  # start tilted, like in swing.py

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
