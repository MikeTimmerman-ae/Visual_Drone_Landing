import numpy as np
from scipy.stats import qmc
import random


class RK4:
    def __init__(self, dxdt, dt, x_dim, u_dim):
        self.dynamics = dxdt
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim

    def step(self, state: np.ndarray, input: np.ndarray) -> np.ndarray:
        """ Discrete-time dynamics (Runge-Kutta 4) of a planar quadrotor """
        assert state.shape == (self.x_dim,), f"{state.shape} does not equal {(self.x_dim,)}"
        assert input.shape == (self.u_dim,), f"{input.shape} does not equal {(self.u_dim,)}"
        # Update state
        k1 = self.dynamics(state, input)
        k2 = self.dynamics(state + self.dt / 2 * k1, input)
        k3 = self.dynamics(state + self.dt / 2 * k2, input)
        k4 = self.dynamics(state + self.dt * k3, input)
        state = state + self.dt * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4)
        return state


def Rot_body2eci(euler_angles: np.ndarray) -> np.ndarray:
    """ Rotation matrix from body frame to inertial frame """
    assert euler_angles.shape == (3, )

    phi = euler_angles[0]
    theta = euler_angles[1]
    psi = euler_angles[2]

    Ryaw = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    Rpitch = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    Rroll = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    Rbe = Ryaw @ Rpitch @ Rroll

    assert Rbe.shape == (3, 3), f"Rotation matrix does not have the correct shape"
    assert np.isclose(np.linalg.norm(Rbe @ np.array([ 1 /np.sqrt(3), 1/ np.sqrt(3), 1 / np.sqrt(3)])), 1), f"Rotation changes vector magnitude"
    return Rbe


def Rot_eci2body(euler_angles: np.ndarray) -> np.ndarray:
    R = Rot_body2eci(euler_angles)
    return R.transpose()


def generate_init_states(num_samples, method):
    bounds = {
        'att': np.pi / 180 * np.array([-1.5, 1.5]),         # rad
        'ang_vel': np.pi / 180 * np.array([-0.0, 0.0]),     # rad/s
        'pos': np.array([[-5, 5],           # xy-pos
                         [75, 100]]),        # z-pos
        'lin_vel': np.array([[-1.0, 1.0],    # xy-vel
                             [-0.5, 0.0]])    # z-vel
    }
    d = 8
    if method == 'uniform':
        # Use random sampling
        samples = np.random.uniform(0, 1, size=(num_samples, d))
    elif method == 'sobol':
        # Use Sobol for largest power of 2 ≤ num_samples, fill remainder with uniform samples
        m = int(np.floor(np.log2(num_samples)))
        sampler = qmc.Sobol(d=d, scramble=True)
        samples_sobol = sampler.random_base2(m=m)  # 2^8 = 256 samples -> (256, 10)
        samples_random = np.random.uniform(0, 1, size=(num_samples - 2**m, d))
        samples = np.concatenate([samples_sobol, samples_random])
    else:
        raise ValueError("Invalid method. Choose 'sobol' or 'uniform'.")

    roll_pitch = (bounds['att'][1] - bounds['att'][0]) * samples[:, :2] + bounds['att'][0]
    yaw = np.zeros((num_samples, 1))

    roll_pitch_rate = np.zeros((num_samples, 2))    # (bounds['ang_vel'][1] - bounds['ang_vel'][0]) * samples[:, 2:4] + bounds['ang_vel'][0]
    yaw_rate = np.zeros((num_samples, 1))

    pos_xy = (bounds['pos'][0, 1] - bounds['pos'][0, 0]) * samples[:, 2:4] + bounds['pos'][0, 0]
    pos_z = np.expand_dims((bounds['pos'][1, 1] - bounds['pos'][1, 0]) * samples[:, 4] + bounds['pos'][1, 0], axis=1)

    vel_xy = (bounds['lin_vel'][0, 1] - bounds['lin_vel'][0, 0]) * samples[:, 5:7] + bounds['lin_vel'][0, 0]
    vel_z = np.expand_dims((bounds['lin_vel'][1, 1] - bounds['lin_vel'][1, 0]) * samples[:, 7] + bounds['lin_vel'][1, 0], axis=1)

    return np.concatenate(
        [roll_pitch, yaw, roll_pitch_rate, yaw_rate, pos_xy, pos_z, vel_xy, vel_z],
        axis=1
    )
