import matplotlib.pyplot as plt
import numpy as np
from infrastructure.replay_buffer import ReplayBuffer
import pickle
import cv2
import time
import matplotlib
# matplotlib.use('TkAgg')

rad2deg = 180 / np.pi
buffer = ReplayBuffer()

with open('agents/expert_data/mpc_1.pkl', 'rb') as f:
    paths = pickle.load(f)
    obs_ls = np.array([path["observation"] for path in paths], dtype=object)
    actions_ls = np.array([path["action"] for path in paths], dtype=object)
    states_ls = np.array([path["state"] for path in paths], dtype=object)

    # obs_img, obs_height, actions_ls, next_obs_img, next_obs_height, terminals, rewards, states_ls = utils.convert_listofrollouts(paths)


# lengths = []
# for ob_height in obs_height:
#     lengths.append(len(ob_height))
# print(sum(lengths))

# Plot camera data
# for rgb_img in obs_img[0]:
#     cv2.imshow("Drone Camera", rgb_img.T)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     time.sleep(0.1)

rgb_img = obs_ls[0][50]['cam']
cv2.imshow("Drone Camera", rgb_img.T)

# # Plot generated paths
# fig1, axs1 = plt.subplots(2, 3, figsize=(12, 3))
# fig2, axs2 = plt.subplots(2, 3, figsize=(12, 3))
# fig3, axs3 = plt.subplots(1, 4, figsize=(12, 3))

# for i in range(len(paths)):
#     actions = actions_ls[i]
#     states = states_ls[i]
#     times = np.linspace(0, len(states) * 0.1, len(states))

#     # Plot angles
#     axs1[0, 0].plot(times, states[:, 0] * rad2deg)
#     axs1[0, 0].set_ylabel('Angular Displacement [deg]')
#     axs1[0, 0].grid()

#     axs1[0, 1].plot(times, states[:, 1] * rad2deg)
#     axs1[0, 1].grid()

#     axs1[0, 2].plot(times, states[:, 2] * rad2deg)
#     axs1[0, 2].grid()

#     # Plot angular velocity
#     axs1[1, 0].plot(times, states[:, 3] * rad2deg)
#     axs1[1, 0].set_ylabel('Angular Velocity [deg/s]')
#     axs1[1, 0].set_xlabel('Time [s]')
#     axs1[1, 0].grid()

#     axs1[1, 1].plot(times, states[:, 4] * rad2deg)
#     axs1[1, 1].set_xlabel('Time [s]')
#     axs1[1, 1].grid()

#     axs1[1, 2].plot(times, states[:, 5] * rad2deg)
#     axs1[1, 2].set_xlabel('Time [s]')
#     axs1[1, 2].grid()

#     # Plot positions
#     axs2[0, 0].plot(times, states[:, 6])
#     axs2[0, 0].set_ylabel('Linear Displacement [m]')
#     axs2[0, 0].grid()

#     axs2[0, 1].plot(times, states[:, 7])
#     axs2[0, 1].grid()

#     axs2[0, 2].plot(times, states[:, 8])
#     axs2[0, 2].grid()

#     # Plot angular velocity
#     axs2[1, 0].plot(times, states[:, 9])
#     axs2[1, 0].set_ylabel('Linear Velocity [m/s]')
#     axs2[1, 0].set_xlabel('Time [s]')
#     axs2[1, 0].grid()

#     axs2[1, 1].plot(times, states[:, 10])
#     axs2[1, 1].set_xlabel('Time [s]')
#     axs2[1, 1].grid()

#     axs2[1, 2].plot(times, states[:, 11])
#     axs2[1, 2].set_xlabel('Time [s]')
#     axs2[1, 2].grid()

#     # Plot inputs
#     axs3[0].plot(times, actions[:, 0])
#     axs3[0].set_xlabel('Motor Speed 1 [rad/s]')
#     axs3[0].set_xlabel('Time [s]')
#     axs3[0].grid()
#     axs3[1].plot(times, actions[:, 1])
#     axs3[1].set_xlabel('Motor Speed 2 [rad/s]')
#     axs3[1].set_xlabel('Time [s]')
#     axs3[1].grid()
#     axs3[2].plot(times, actions[:, 2])
#     axs3[2].set_xlabel('Motor Speed 3 [rad/s]')
#     axs3[2].set_xlabel('Time [s]')
#     axs3[2].grid()
#     axs3[3].plot(times, actions[:, 3])
#     axs3[3].set_xlabel('Motor Speed 4 [rad/s]')
#     axs3[3].set_xlabel('Time [s]')
#     axs3[3].grid()

# plt.show()
