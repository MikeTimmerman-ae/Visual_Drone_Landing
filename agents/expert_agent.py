import cvxpy as cvx
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('TkAgg')


class MPC:
    def __init__(self, config):
        self.xf = np.zeros((12, ))

        # Controller parameters
        self.dt = config.expert_config.dt
        self.tN = config.expert_config.tN
        self.N = round(self.tN / self.dt)
        self.tM = config.expert_config.tM
        self.M = round(self.tM / self.dt)
        self.R = config.expert_config.R
        self.P = config.expert_config.P

        # Quadcopter and environment parameters
        self.env_dt = config.env_config.dt
        self.g = config.env_config.g
        self.m = config.drone_config.m
        self.lx = config.drone_config.lx
        self.ly = config.drone_config.ly
        self.I = jnp.array(config.drone_config.I)
        self.kf = config.drone_config.kf
        self.km = config.drone_config.km
        self.max_rotor_speed = np.sqrt(np.abs(4 * self.m * self.g / (4 * self.kf)))     # rad/s
        self.max_rotor_acc = 500                                                        # rad/s^2

        # Control buffer management
        self.j = 0
        self.input_buffer = None
        self.sample_ratio = round(self.dt / self.env_dt)

        # Linearize dynamics around steady-state hover
        self.x_hover = jnp.zeros((12,))
        self.u_hover = jnp.sqrt(self.m * self.g / (4 * self.kf)) * jnp.ones((4,))
        self.A = jnp.diag(np.ones((12,))) + self.dt * jax.jacobian(lambda x: self.f(x, self.u_hover))(self.x_hover)
        self.B = self.dt * jax.jacobian(lambda u: self.f(self.x_hover, u))(self.u_hover)

    def reset(self):
        self.j = 0
        self.input_buffer = None

    def get_action(self, t, x, u):
        if self.input_buffer is None or self.j == round(self.tM / self.env_dt):
            self.j = 0
            self.input_buffer = self.get_actions(t, x, u)
        action = self.input_buffer[self.j, :]
        self.j += 1
        return action

    def get_actions(self, t0, x0, u0):
        _, _, u, _ = self.LTI_mpc(t0, x0, u0)
        actions = np.repeat(u[:self.M, :], self.sample_ratio, axis=0)
        return actions

    def LTI_mpc(self, t0, x0, u0):
        # Time vector
        tf = t0 + self.N * self.dt
        ts = np.linspace(0, tf, self.N)

        # Solve optimization problem
        x_cvx = cvx.Variable((self.N + 1, 12))
        u_cvx = cvx.Variable((self.N, 4))

        # Build up the objective function
        if np.linalg.norm(x0[6:9] - self.xf[6:9]) <= 10:
            objective = 0
            for k in range(self.N):
                objective += cvx.atoms.quad_form(u_cvx[k, :], self.R)  # input cost
        else:
            objective = cvx.atoms.quad_form(x_cvx[self.N] - self.xf, self.P)            # final state cost
            for k in range(self.N):
                objective += cvx.atoms.quad_form(u_cvx[k, :] - self.u_hover, self.R)  # input cost

        # Build up the constrains
        constraints = []
        # Initial and final constraint
        constraints.append(x_cvx[0] == x0)
        if np.linalg.norm(x0[6:9] - self.xf[6:9]) <= 10:
            constraints.append(x_cvx[self.N] == self.xf)  # final constraint
        # dynamics constraint
        constraints.append(self.A @ (x_cvx[:-1, :] - self.x_hover).T + self.B @ (u_cvx - self.u_hover).T == x_cvx[1:, :].T)
        # Input constraints
        constraints.append(u_cvx >= 0)
        constraints.append(u_cvx <= self.max_rotor_speed)
        constraints.append((u_cvx[1, :] - u0) / self.dt <= self.max_rotor_acc)
        constraints.append((u_cvx[1:, :] - u_cvx[:-1, :]) / self.dt <= self.max_rotor_acc)

        # state constraints
        constraints.append(x_cvx[:, 8] >= 0.)                       # Z-position
        constraints.append(x_cvx[:, :2] <= 30 * np.pi / 180)        # Attitude roll and pitch angles

        # Solve optimization problem
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()
        if prob.status != "optimal":
            raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
        x = x_cvx.value
        u = u_cvx.value
        J = prob.objective.value
        return ts, x, u, J

    def f(self, state, inputs):
        dx = jnp.zeros((12,))

        Reb_ = self.Reb(state[0:3])
        # Forces
        thrust = Reb_ @ jnp.array([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [self.kf, self.kf, self.kf, self.kf]]) @ inputs ** 2
        gravity = jnp.array([0, 0, -self.m * self.g])
        force = thrust + gravity

        # Get external moment acting on drone
        control_moment = jnp.array([[-self.lx * self.kf, -self.lx * self.kf, self.lx * self.kf, self.lx * self.kf],
                                   [self.ly * self.kf, -self.ly * self.kf, -self.ly * self.kf, self.ly * self.kf],
                                   [self.km, -self.km, self.km, -self.km]]) @ inputs ** 2
        moment = control_moment

        # Derivatives of attitude angles (phi, theta, psi)
        kinematics = jnp.array([[1, jnp.tan(state[1]) * jnp.sin(state[0]), jnp.tan(state[1]) * jnp.cos(state[0])],
                                [0, jnp.cos(state[0]), -jnp.sin(state[0])],
                                [0, jnp.sin(state[0]) / jnp.cos(state[1]), jnp.cos(state[0]) / jnp.cos(state[1])]])
        dx = dx.at[0:3].set(kinematics @ state[3:6])  # dphi, dtheta, dpsi

        # Derivative of angular velocity (p, q, r)
        dx = dx.at[3:6].set(jnp.array([((self.I[1, 1] - self.I[2, 2]) * state[5] * state[4] + moment[0]) / self.I[0, 0],
                                       ((self.I[2, 2] - self.I[0, 0]) * state[5] * state[3] + moment[1]) / self.I[1, 1],
                                       ((self.I[0, 0] - self.I[1, 1]) * state[4] * state[3] + moment[2]) / self.I[2, 2]]))

        # Derivative of position (x, y, z)
        dx = dx.at[6:9].set(state[9:12])                                        # dx, dy, dz

        # Derivative of velocity (vx, vy, vz)
        dx = dx.at[9:12].set(force / self.m)                                  # dvx, dvy, dvz

        return dx

    def Reb(self, euler_angles: np.ndarray) -> jnp.ndarray:
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]

        Ryaw = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                         [jnp.sin(psi), jnp.cos(psi), 0],
                         [0, 0, 1]])
        Rpitch = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                           [0, 1, 0],
                           [-jnp.sin(theta), 0, jnp.cos(theta)]])
        Rroll = jnp.array([[1, 0, 0],
                          [0, jnp.cos(phi), -jnp.sin(phi)],
                          [0, jnp.sin(phi), jnp.cos(phi)]])
        Reb = Ryaw @ Rpitch @ Rroll

        return Reb