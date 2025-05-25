import random
import torch
import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from config import Config
from environment.quadcopter import Quadcopter
from environment.pad import create_pad


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, **kwargs):
        self.config: Config = Config()

        # Environment variables
        self.i_step = int(0)
        self.time = self.config.env_config.t0
        self.dt = self.config.env_config.dt

        self.max_ep_time = self.config.training.max_ep_time
        self.landing_height = self.config.training.landing_height
        self.max_steps = int(self.max_ep_time / self.dt)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)    # Normalized PD gains
        self.observation_space = spaces.Dict({
            'cam': spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8),
            'radar': spaces.Box(low=0.0, high=250.0, shape=(1,), dtype=np.float32)
        })

        # Create PyBullet environment
        self.render = kwargs['render']
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        create_pad()
        self.quad: Quadcopter = Quadcopter(self.config)

        # Log environment variables
        self.reach_count = 0
        self.deviation_count = 0
        self.timeout_count = 0

        # Logging
        self.times = np.zeros((self.max_steps,))
        self.states = np.zeros((self.max_steps, 12))
        self.actions = np.zeros((self.max_steps, 4))

        print("[INFO] Finished setting up Environement")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_obs(self):
        rgba_img = self.quad.get_camera_image()
        rgb_img = rgba_img[:, :, :3].T
        height = np.array([self.quad.position[2]])
        return {'cam': rgb_img,
                'radar': np.float32(height)}

    def _get_reward(self):
        return 0

    def _get_termination(self):
        # Terminate episode if drone has landed or max time has been reached
        return self.quad.position[2] < self.landing_height or self.time > self.max_ep_time

    def _get_info(self):
        return {'curr_time': self.time,
                'curr_state': self.quad.state,
                'curr_action': self.quad.input,
                'curr_step': self.i_step,
                'reach_count': self.reach_count,
                'deviation_count': self.deviation_count,
                'timeout_count': self.timeout_count,
                'times': self.times[:self.i_step+1],
                'states': self.states[:self.i_step+1, :],
                'actions': self.actions[:self.i_step, :]}

    def reset(self, *, seed=None, options=None):
        # Reset environment
        super().reset(seed=seed)        # seed self.np_random
        if seed is not None:
            self.set_seed(seed)

        if options is None or options['init_state'] is None:
            init_att = self.config.traj_config.init_att
            init_ang_vel = self.config.traj_config.init_ang_vel
            init_pos = self.config.traj_config.init_pos
            init_lin_vel = self.config.traj_config.init_lin_vel
            state = np.hstack((init_att, init_ang_vel, init_pos, init_lin_vel))
        else:
            state = options['init_state']
        self.quad.reset(state)

        self.i_step = 0
        self.time = self.config.env_config.t0
        self.times[self.i_step] = self.time
        self.states[self.i_step, :] = self.quad.state

        # Get initial observation and info
        obs = self._get_obs()
        info = self._get_info()

        if options is not None and options['return_info']:
            return obs, info
        return obs

    def step(self, action):
        assert action.shape == (4,), "Action should be size (4,)"

        # Update drone dynamics according to control input
        self.time += self.dt
        self.quad.step(action)

        # Log progress
        self.i_step += 1
        self.times[self.i_step] = self.time
        self.states[self.i_step, :] = self.quad.state
        self.actions[self.i_step-1, :] = action

        # Return properties
        obs = self._get_obs()
        reward = self._get_reward()
        termination = self._get_termination()
        info = self._get_info()

        return obs, reward, termination, False, info
