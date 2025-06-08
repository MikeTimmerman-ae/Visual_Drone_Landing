"""
READ-ONLY: Behavior cloning agent definition
"""
from infrastructure.replay_buffer import ReplayBuffer
from policies.CNN_LSTM_policy import PolicyNetwork
from .base_agent import BaseAgent
import infrastructure.pytorch_util as ptu
import numpy as np


class BCAgent(BaseAgent):
    """
    Attributes
    ----------
    actor : MLPPolicySL
        An MLP that outputs an agent's actions given its observations
    replay_buffer: ReplayBuffer
        A replay buffer which stores collected trajectories

    Methods
    -------
    train:
        Calls the actor update function
    add_to_replay_buffer:
        Updates a the replay buffer with new paths
    sample
        Samples a batch of trajectories from the replay buffer
    """
    def __init__(self, agent_params):
        super(BCAgent, self).__init__()

        # Initialize variables
        self.agent_params = agent_params

        # Create policy class as our actor
        self.actor = PolicyNetwork(
            self.agent_params['ac_dim'],
            learning_rate=self.agent_params['learning_rate'],
        )

        if 'full_policy' in agent_params and agent_params['full_policy'] is not None:
            self.actor.load(agent_params['full_policy'])
        if 'cnn_policy' in agent_params and agent_params['cnn_policy'] is not None:
            self.actor.load_cnn(agent_params['cnn_policy'])
        if 'lstm_policy' in agent_params and agent_params['lstm_policy'] is not None:
            self.actor.load_lstm(agent_params['lstm_policy'])

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.agent_params['replay_buffer_seq_len'],
            self.agent_params['replay_buffer_max_ep']
        )

    def reset(self):
        self.actor.reset()

    def get_action(self, image_ob: np.ndarray, state_ob: np.ndarray, use_cnn=False) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        
        # Pre-process observations
        image_ob, state_ob = self.process_obs(image_ob, state_ob)

        # Return the action that the policy prescribes
        dist, _ = self.actor.forward(image_ob, state_ob, use_cnn)
        action = ptu.to_numpy(dist.mean)
        action = np.array([
            12.5 * (action[0, 0] + 1.),
            10e-3 * action[0, 1],
            10e-3 * action[0, 2],
            10e-3 * action[0, 3],
        ])
        return action

    def train(self, img_seq, state_seq, ac_seq, use_cnn=False):
        """
        :param img_seq: batch_size x time_steps x obs_dim batch of observations
        :param state_seq: batch_size x time_steps x obs_dim batch of observations
        :param ac_seq: batch_size x time_steps x ac_dim batch of actions
        """

        # Training a behaviour cloning agent refers to updating its actor using
        # the given observations and corresponding action labels
        T = img_seq.shape[1]
        self.actor.reset()
        total_loss = 0
        for t in range(T):
            # Update policy
            image_ob, state_ob = self.process_obs(img_seq[:, t], state_seq[:, t])
            ac = self.process_action(ac_seq[:, t])
            log = self.actor.update(image_ob, state_ob, ac, **{'use_cnn': use_cnn})
            total_loss += log['Training Loss']
        return {
            'Training Loss': total_loss / T,
        }
    
    def train_cnn(self, img_ob, state_label):
        """
        :param img_ob: batch_size x obs_dim batch of observations
        :param state_label: batch_size x n_state batch of states
        """
        # Update policy
        img_ob = img_ob.reshape(-1, *img_ob.shape[2:])
        state_label = state_label.reshape(-1, state_label.shape[-1])

        img_ob, state_label = self.process_obs(img_ob, state_label)
        log = self.actor.update_cnn(img_ob, state_label)
        return log
    
    def eval_cnn(self, img_ob, state_label):
        """
        :param img_ob: batch_size x obs_dim batch of observations
        :param state_label: batch_size x n_state batch of states
        """
        # Update policy
        image_ob, state_label = self.process_obs(img_ob, state_label)
        log = self.actor.update_cnn(image_ob, state_label, eval=True)
        return log['Training Loss']

    def add_to_replay_buffer(self, paths):
        """
        :param paths: paths to add to the replay buffer
        """
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        :param batch_size: size of batch to sample from replay buffer
        """
        # HW1: you will modify this
        return self.replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        """
        :param path: path to save
        """
        return self.actor.save(path)
    
    def save_lstm(self, path):
        """ Save LSTM weights
        :param path: path to save
        """
        return self.actor.save_lstm(path)

    def save_cnn(self, path):
        """ Save CNN weights
        :param path: path to save
        """
        return self.actor.save_cnn(path)
    
    def process_obs(self, image_ob: np.ndarray, state_ob: np.ndarray):

        # Expand over batch dimension if necessary
        if image_ob.ndim == 3:  # (C, H, W)
            image_ob = np.expand_dims(image_ob, axis=0)        # → (1, C, H, W)
        if state_ob.ndim == 1:  # (1,)
            state_ob = np.expand_dims(state_ob, axis=0)        # → (1, 1)

        # Normalize image and convert to gray-scale
        image_rgb_ob_norm = image_ob.astype(np.float32) / 255.
        weights = np.array([0.1, 0.7, 0.2], dtype=np.float32).reshape(3, 1, 1)
        image_gray_ob_norm = np.sum(image_rgb_ob_norm * weights, axis=1, keepdims=True)
        
        # Normalize state observation
        state_ob = np.concatenate([
            state_ob[:, 0:3],               # (pitch/roll/yaw)
            state_ob[:, 3:6],               # (pitch/roll/yaw rate)
            state_ob[:, 6:8] / 10,          # (xy-pos)
            state_ob[:, 8:9] / 150,         # (z-pos)
            state_ob[:, 9:12],              # (xyz-vel)
        ], axis=1)

        return ptu.from_numpy(image_gray_ob_norm), ptu.from_numpy(state_ob)
    
    def process_action(self, actions: np.ndarray):

        # Normalize actions
        ac_norm = np.column_stack((actions[:, 0] / 12.5 - 1.0,
                                   actions[:, 1] / 10e-3,
                                   actions[:, 2] / 10e-3,
                                   actions[:, 3] / 10e-3))

        return ptu.from_numpy(ac_norm)
