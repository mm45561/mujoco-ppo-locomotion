import mujoco

model = mujoco.MjModel.from_xml_path("02-mujoco-pendulum/pendulum.xml")
data = mujoco.MjData(model)

data.ctrl[0] = 0.5  # target angle for the motor, in radians

for step in range(1000):
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        print(f"t={data.time:.2f}s  target=0.5  angle={data.qpos[0]:.3f} rad")
