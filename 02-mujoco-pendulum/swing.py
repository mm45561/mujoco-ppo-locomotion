import mujoco

model = mujoco.MjModel.from_xml_path("02-mujoco-pendulum/pendulum.xml")
data = mujoco.MjData(model)

data.qpos[0] = 1.0  # start tilted by 1 radian (about 57 degrees)

for step in range(500):
    mujoco.mj_step(model, data)
    if step % 5 == 0:
        print(f"t={data.time:.2f}s  angle={data.qpos[0]:.3f} rad")
