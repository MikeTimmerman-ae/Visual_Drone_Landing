"""
READ-ONLY: Behavior cloning agent definition
"""
from infrastructure.replay_buffer import ReplayBuffer
from policies.CNN_LSTM_policy import PolicyNetwork
from .base_agent import BaseAgent
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
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # Initialize variables
        self.env = env
        self.agent_params = agent_params

        # Create policy class as our actor
        self.actor = PolicyNetwork(
            self.agent_params['ac_dim'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.agent_params['replay_buffer_seq_len'],
            self.agent_params['replay_buffer_max_ep']
        )

    def reset(self):
        self.actor.reset()

    def train(self, img_seq, height_seq, ac_na):
        """
        :param img: batch_size x time_steps x obs_dim batch of observations
        :param height: batch_size x time_steps x obs_dim batch of observations
        :param ac_na: batch_size x time_steps x ac_dim batch of actions
        """
        # Normalized actions and observations
        img_seq_norm = img_seq.astype(np.float32) / 255
        height_seq_norm = height_seq / 150
        ac_norm = (ac_na - 40)  / 80

        # Training a behaviour cloning agent refers to updating its actor using
        # the given observations and corresponding action labels
        T = img_seq.shape[1]
        self.actor.reset()
        total_loss = 0
        for t in range(T):
            # Update policy
            log = self.actor.update(img_seq_norm[:, t], height_seq_norm[:, t], ac_norm[:, t])
            total_loss += log['Training Loss']
        return {
            'Training Loss': total_loss,
        }

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
