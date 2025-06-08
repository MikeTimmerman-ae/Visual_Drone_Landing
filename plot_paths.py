import numpy as np
from infrastructure.replay_buffer import ReplayBuffer
import pickle
import time
from environment.quadcopter import Quadcopter
from config import Config
import pybullet as p
from environment.pad import create_pad

rad2deg = 180 / np.pi
buffer = ReplayBuffer()

paths = []
for i in range(6):
    with open(f"agents/expert_data/mpc_{i+1}.pkl", 'rb') as f:
        paths += pickle.load(f)

# with open(f"agents/expert_data/eval_policy_0.pkl", 'rb') as f:
#     paths = pickle.load(f)


# p.connect(p.DIRECT)
# p.setGravity(0, 0, 0)
# p.setRealTimeSimulation(0)
# create_pad()
# quad = Quadcopter(Config())
# for i, path in enumerate(paths):
#     for j, state in enumerate(path["state"]):
#         quad.reset(state)
#         rgba_img = quad.get_camera_image()
#         rgb_img = rgba_img[:, :, :3].T
#         paths[i]["ob_image"][j] = rgb_img

# for i in range(6):
#     with open(f"agents/expert_data/mpc_{i+1}.pkl", "wb") as f:
#         pickle.dump(paths[50*i:50*(i+1)], f)


obs_ls = np.array([path["ob_image"] for path in paths], dtype=object)
actions_ls = np.array([path["action"] for path in paths], dtype=object)
states_ls = np.array([path["state"] for path in paths], dtype=object)
reward_ls = np.array([path["reward"] for path in paths], dtype=object)
            
train_returns = [path["reward"].sum() for path in paths]
train_ep_lens = [len(path["reward"]) for path in paths]

print(np.mean(train_returns))
print(np.std(train_returns))
print(np.mean(train_ep_lens))

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

# rgb_img = obs_ls[0][50]['cam']
# cv2.imshow("Drone Camera", rgb_img.T)

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
