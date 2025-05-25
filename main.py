import pybullet as p
import pybullet_data
import cv2
import time
import numpy as np
from agents.expert_agent import MPC
from policies.CNN_LSTM_policy import PolicyNetwork
import matplotlib.pyplot as plt

from environment.quadcopter import Quadcopter
from environment.pad import create_pad
from config import Config

rad2deg = 180 / np.pi
render = True

# Connect to GUI
if render:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load landing pad
create_pad()

# Configure drone and expert policy
config = Config()
drone = Quadcopter(config)
expert = MPC(config)
agent = PolicyNetwork(ac_dim=4)

# Simulate
time0, state0, inputs0, J = expert.LTI_mpc(0., drone.state, np.zeros((4,)))
states = [state0[0, :]]
actions = []
times = []

t = 0
t0 = time.time()
while drone.position[2] > 0.1:
    # Get observation
    rgba_img = drone.get_camera_image()
    rgb_img = rgba_img[:, :, :3].T
    height = np.array([drone.position[2]])

    # Get action
    action = expert.get_action(t, drone.state)

    # Step environment
    drone.step(action)
    times.append(t)
    states.append(drone.state)
    actions.append(action)
    t = t + config.env_config.dt

    if render:
        cv2.imshow("Drone Camera", rgba_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f'Finished in {time.time() - t0} sec')
states = np.vstack(states)
actions = np.vstack(actions)

# Plots
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.view_init(20, -45)
ax1.set_xlabel("X-position [m]")
ax1.set_ylabel("Y-position [m]")
ax1.set_zlabel("Z-position [m]")
ax1.plot(state0[:, 6], state0[:, 7], state0[:, 8], label='Planned Trajectory')
ax1.plot(states[:, 6], states[:, 7], states[:, 8], label='Executed Trajectory')
ax1.legend()

fig2, axs = plt.subplots(2, 3, figsize=(12, 3))
# Plot angles
axs[0, 0].plot(time0, state0[:-1, 0] * rad2deg, label='Planned States')
axs[0, 0].plot(times, states[:-1, 0] * rad2deg, label='True States')
axs[0, 0].set_ylabel('Angular Displacement [deg]')
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(time0, state0[:-1, 1] * rad2deg)
axs[0, 1].plot(times, states[:-1, 1] * rad2deg)
axs[0, 1].grid()

axs[0, 2].plot(time0, state0[:-1, 2] * rad2deg)
axs[0, 2].plot(times, states[:-1, 2] * rad2deg)
axs[0, 2].grid()

# Plot angular velocity
axs[1, 0].plot(time0, state0[:-1, 3] * rad2deg)
axs[1, 0].plot(times, states[:-1, 3] * rad2deg)
axs[1, 0].set_ylabel('Angular Velocity [deg/s]')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].grid()

axs[1, 1].plot(time0, state0[:-1, 4] * rad2deg)
axs[1, 1].plot(times, states[:-1, 4] * rad2deg)
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].grid()

axs[1, 2].plot(time0, state0[:-1, 5] * rad2deg)
axs[1, 2].plot(times, states[:-1, 5] * rad2deg)
axs[1, 2].set_xlabel('Time [s]')
axs[1, 2].grid()

fig3, axs = plt.subplots(2, 3, figsize=(12, 3))
# Plot positions
axs[0, 0].plot(time0, state0[:-1, 6], label='Planned States')
axs[0, 0].plot(times, states[:-1, 6], label='True States')
axs[0, 0].set_ylabel('Linear Displacement [m]')
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(time0, state0[:-1, 7])
axs[0, 1].plot(times, states[:-1, 7])
axs[0, 1].grid()

axs[0, 2].plot(time0, state0[:-1, 8])
axs[0, 2].plot(times, states[:-1, 8])
axs[0, 2].grid()

# Plot angular velocity
axs[1, 0].plot(time0, state0[:-1, 9])
axs[1, 0].plot(times, states[:-1, 9])
axs[1, 0].set_ylabel('Linear Velocity [m/s]')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].grid()

axs[1, 1].plot(time0, state0[:-1, 10])
axs[1, 1].plot(times, states[:-1, 10])
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].grid()

axs[1, 2].plot(time0, state0[:-1, 11])
axs[1, 2].plot(times, states[:-1, 11])
axs[1, 2].set_xlabel('Time [s]')
axs[1, 2].grid()

# Plot inputs
fig4, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].plot(time0, inputs0[:, 0], label='Planned Inputs')
axs[0].plot(times, actions[:, 0], label='Executed Inputs')
axs[0].legend()
axs[1].plot(time0, inputs0[:, 1])
axs[1].plot(times, actions[:, 1])
axs[2].plot(time0, inputs0[:, 2])
axs[2].plot(times, actions[:, 2])
axs[3].plot(time0, inputs0[:, 3])
axs[3].plot(times, actions[:, 3])

plt.show()

p.disconnect()
cv2.destroyAllWindows()
