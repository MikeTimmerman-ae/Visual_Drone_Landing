import pybullet as p
import numpy as np
from infrastructure.dynamics_utils import Rot_body2eci, Rot_eci2body
from infrastructure.dynamics_utils import RK4


class Quadcopter:
    def __init__(self, config):
        # Configuration
        self.config = config
        self.x_dim = config.drone_config.x_dim
        self.u_dim = config.drone_config.u_dim

        # State variables
        init_att = config.traj_config.init_att
        init_ang_vel = config.traj_config.init_ang_vel
        init_pos = config.traj_config.init_pos
        init_lin_vel = config.traj_config.init_lin_vel
        self.state = np.hstack((init_att, init_ang_vel, init_pos, init_lin_vel))
        self.input = np.zeros((4,))
        self.dt = config.env_config.dt
        self.propagator = RK4(self.dxdt, self.dt, self.x_dim, self.u_dim)

        # Configuration variables
        self.m = config.drone_config.m
        self.I = config.drone_config.I
        self.lx = config.drone_config.lx
        self.ly = config.drone_config.ly
        self.lz = config.drone_config.ly
        self.rho = config.env_config.rho
        self.g = config.env_config.g
        self.kf = config.drone_config.kf
        self.km = config.drone_config.km
        self.A = config.drone_config.A
        self.Cd_v = config.drone_config.Cd_v
        self.Cd_om = config.drone_config.Cd_om
        self.max_rotor_speed = np.sqrt(np.abs(4 * self.m * self.g / (4 * self.kf)))
        self.id = self.create_id()
        p.changeDynamics(self.id, -1, linearDamping=0, angularDamping=0)

    @property
    def attitude(self):
        return self.state[0:3]

    @property
    def ang_velocity(self):
        return self.state[3:6]

    @property
    def position(self):
        return self.state[6:9]

    @property
    def lin_velocity_b(self):
        return Rot_eci2body(self.attitude) * self.lin_velocity_e

    @property
    def lin_velocity_e(self):
        return self.state[9:12]

    def create_id(self):
        drone_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX, halfExtents=[self.lx * 2, self.ly * 2, self.lz]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX, halfExtents=[self.lx * 2, self.ly * 2, self.lz],
                rgbaColor=[1, 1, 1, 1]
            ),
            basePosition=self.position
        )
        return drone_id

    def reset(self, state0):
        self.input = np.zeros((4,))
        self.state = state0
        # Update visualization
        p.resetBasePositionAndOrientation(self.id,
                                          posObj=self.position,
                                          ornObj=p.getQuaternionFromEuler(self.attitude))
        p.resetBaseVelocity(self.id, linearVelocity=self.lin_velocity_e, angularVelocity=self.ang_velocity)

    def step(self, action: np.ndarray):
        # Propagate drone state
        self.input = action
        self.state = self.propagator.step(self.state, action)
        if self.position[2] <= 0:
            self.state[8] = 0.0
            self.state[11] = 0.0

        # Update visualization
        p.resetBasePositionAndOrientation(self.id,
                                          posObj=self.position,
                                          ornObj=p.getQuaternionFromEuler(self.attitude))
        p.resetBaseVelocity(self.id, linearVelocity=self.lin_velocity_e, angularVelocity=self.ang_velocity)

    def dxdt(self, state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        state: [phi, theta, psi, p, q, r, x, y, z, vx, vy, vz]
        """
        # Decompose state
        att = state[0:3]
        omega = state[3:6]
        vel = state[9:12]

        dx = np.zeros((self.x_dim,))
        inputs = np.clip(inputs, 0, self.max_rotor_speed)
        R_body2eci = Rot_body2eci(att)

        # Get external force acting on drone (in ECI frame)
        thrust = R_body2eci @ np.array([[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [self.kf, self.kf, self.kf, self.kf]]) @ inputs ** 2
        gravity = np.array([0, 0, -self.m * self.g])
        drag = self.Cd_v * 1 / 2 * self.rho * vel ** 2 * self.A
        force = thrust + gravity + drag
        if self.position[2] <= 0.0 and force[2] <= 0.0:
            force[2] = 0.0

        # Get external moment acting on drone
        control_moment = np.array([[-self.lx * self.kf, -self.lx * self.kf, self.lx * self.kf, self.lx * self.kf],
                                   [self.ly * self.kf, -self.ly * self.kf, -self.ly * self.kf, self.ly * self.kf],
                                   [self.km, -self.km, self.km, -self.km]]) @ inputs ** 2
        moment = control_moment

        # Derivatives of attitude angles (phi, theta, psi)
        kinematics = np.array([[1, np.tan(att[1]) * np.sin(att[0]), np.tan(att[1]) * np.cos(att[0])],
                               [0, np.cos(att[0]), -np.sin(att[0])],
                               [0, np.sin(att[0]) / np.cos(att[1]), np.cos(att[0]) / np.cos(att[1])]])
        dx[0:3] = kinematics @ omega  # dphi, dtheta, dpsi

        # Derivative of angular velocity (p, q, r)
        dx[3:6] = np.array([((self.I[1, 1] - self.I[2, 2]) * omega[2] * omega[1] + moment[0]) / self.I[0, 0],
                            ((self.I[2, 2] - self.I[0, 0]) * omega[2] * omega[0] + moment[1]) / self.I[1, 1],
                            ((self.I[0, 0] - self.I[1, 1]) * omega[1] * omega[0] + moment[2]) / self.I[2, 2]])

        # Derivative of position (x, y, z)
        dx[6:9] = vel  # dx, dy, dz

        # Derivative of velocity (vx, vy, vz)
        dx[9:12] = force / self.m  # dvx, dvy, dvz

        return dx

    def get_camera_image(self, width=128, height=128, fov=45, near=0.01, far=500):
        pos, ori = p.getBasePositionAndOrientation(self.id)
        rot_matrix = p.getMatrixFromQuaternion(ori)

        # Local -Z is the third column of rotation matrix
        down_vec = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        camera_eye = np.array(pos)
        camera_target = camera_eye - 0.1 * down_vec  # small step downward

        # Local +Y is the second column of rotation matrix -> up vector
        up_vec = np.array([rot_matrix[3], rot_matrix[4], rot_matrix[5]])

        view_matrix = p.computeViewMatrix(camera_eye, camera_target, up_vec)
        projection_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

        _, _, px, _, _ = p.getCameraImage(width=width, height=height,
                                          viewMatrix=view_matrix,
                                          projectionMatrix=projection_matrix,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgba_img = np.array(px, dtype=np.uint8).reshape((height, width, 4))
        return rgba_img
