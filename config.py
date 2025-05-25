import numpy as np
import torch


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    # Configuration of Environment
    env_config = BaseConfig()
    env_config.env_name = 'environment:environment/FlightArena'
    env_config.dt = 0.1
    env_config.t0 = 0.
    env_config.seed = 10
    env_config.g = 10.0  # gravity (m / s**2)
    env_config.rho = 1.225  # gravity (kg / m**3)

    # Training configurations
    training = BaseConfig()
    training.num_processes = 3
    training.num_threads = 1
    training.num_env_steps = 1.2e5
    training.max_ep_time = 30
    training.landing_height = 0.5

    training.continue_training = True
    training.model_dir = 'data/policy-v6/'
    training.new_model = 'model-v1'
    training.new_policy = 'policy-v1'
    training.parent_model = 'model-v0'    #'/best_policy/model'
    training.overwrite = False
    training.log_interval = 5

    training.no_cuda = True  # disables CUDA training
    training.cuda = not training.no_cuda and torch.cuda.is_available()
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)

    # PPO configurations
    ppo = BaseConfig()
    ppo.num_steps = 2048

    # Configuration of expert agent
    expert_config = BaseConfig()
    expert_config.dt = 0.1
    expert_config.tN = 5.0                                  # planning horizon
    expert_config.tM = 0.1                                  # control horizon
    expert_config.R = np.diag([1e-1, 1e-1, 1e-1, 1e-1])     # Input cost
    expert_config.P = 1e1 * np.diag(np.ones((12,)))         # Terminal state cost

    # Configuration of drone
    drone_config = BaseConfig()
    drone_config.x_dim = 12  # state dimension (see dynamics below)
    drone_config.u_dim = 4  # control dimension (see dynamics below)
    drone_config.m = 1.5  # mass (kg)
    drone_config.lx = 0.25 / 2  # half-length (m)
    drone_config.ly = 0.25 / 2  # half-width (m)
    drone_config.lz = 0.1       # height (m)
    drone_config.kf = 3.5e-4      # force constant (N s^2)
    drone_config.km = 1e-5      # moment constant (Nm s^2)
    drone_config.I = np.diag([9.0625e-3, 9.0625e-3, 1.563e-2])  # moment of inertia about the out-of-plane axis (kg * m**2)
    drone_config.A = 0.025  # effective area (m2)
    drone_config.Cd_v = 0.3  # translational drag coefficient
    drone_config.Cd_om = 0.02255  # rotational drag coefficient

    # Configuration of randomly generated trajectory
    traj_config = BaseConfig()
    traj_config.init_att = np.array([-5.0, 5.0, 0.0]) * np.pi / 180
    traj_config.init_ang_vel = np.array([3.0, 3.0, 0.0]) * np.pi / 180
    traj_config.init_pos = np.array([25.0, 25.0, 125.0])
    traj_config.init_lin_vel = np.array([1.5, 1.5, -1.0])
    traj_config.tf = 60
    traj_config.min_dist = 25
    traj_config.max_dist = 50
    traj_config.v_des = 6
