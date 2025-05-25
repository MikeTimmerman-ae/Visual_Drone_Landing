"""
TO EDIT: Defines a pytorch policy as the agent's actor.

"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu
from policies.base_policy import BasePolicy


class PolicyNetwork(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # Initialize variables for environment (action/observation dimension, number of layers, etc.)
        self.ac_dim = ac_dim
        self.learning_rate = learning_rate

        # Policy Modules
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),  # -> [32, 126, 126]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),  # -> [64, 61, 61]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),  # -> [128, 29, 29]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),  # -> [256, 13, 13]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → [256, 1, 1]
            nn.Flatten(),  # → [256]
            nn.Linear(256, 128),  # Drastically fewer params
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=129, hidden_size=256, num_layers=1, batch_first=True)
        self.hidden_state = None

        self.mean_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.ac_dim),  # 4 motor speeds
            nn.Tanh()
        )

        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )

        self.to(ptu.device)

        # Parameter optimizer
        self.optimizer = optim.Adam(
            itertools.chain(
                [self.logstd],
                self.cnn.parameters(),
                self.lstm.parameters(),
                self.mean_net.parameters(),
            ),
            lr=self.learning_rate
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def reset(self):
        """ Reset hidden states """
        self.hidden_state = None

    ##################################

    def get_action(self, image_seq: np.ndarray, height_seq: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        image_seq = ptu.from_numpy(image_seq)
        height_seq = ptu.from_numpy(height_seq)

        if image_seq.ndim == 3:  # (C, H, W)
            image_seq = image_seq.unsqueeze(0)  # → (1, C, H, W)
        if height_seq.ndim == 1:  # (1,)
            height_seq = height_seq.unsqueeze(0)  # → (1, 1)

        # Return the action that the policy prescribes
        dist = self.forward(image_seq, height_seq)
        action = dist.mean
        action = 80 * ptu.to_numpy(action.detach()) + 40
        return action

    def forward(self, image_seq: torch.FloatTensor, height_seq: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param image_seq:  (B, 3, 256, 256) image batch
        :param height_seq: (B, 1) scalar features
        :return:
            action: sampled action(s) from the policy
        """

        # Merge batch and time to encode images
        cnn_out = self.cnn(image_seq)  # (B, 128)

        # Concatenate height
        lstm_input = torch.cat([cnn_out, height_seq], dim=-1).unsqueeze(dim=1)  # (B, 1, 129)

        # Pass through LSTM
        if self.hidden_state is not None:
            h, c = self.hidden_state
            self.hidden_state = (h.detach(), c.detach())
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)  # (B, 1, 256)

        # Use last output to predict action
        mean = self.mean_net(lstm_out[:, -1])  # (B, ac_dim)
        std = torch.exp(self.logstd)
        return distributions.Normal(mean, std)

    def update(self, image_seq: np.ndarray, height_seq: np.ndarray, actions: np.ndarray, **kwargs):
        """
        Updates/trains the policy

        :param image_seq:  (B, 3, 256, 256) image batch
        :param height_seq: (B, 1) scalar features
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """

        actions = ptu.from_numpy(actions)
        image_seq = ptu.from_numpy(image_seq)
        height_seq = ptu.from_numpy(height_seq)

        if image_seq.ndim == 3:  # (C, H, W)
            image_seq = image_seq.unsqueeze(0)  # → (1, C, H, W)
        if height_seq.ndim == 1:  # (1,)
            height_seq = height_seq.unsqueeze(0)  # → (1, 1)

        # Compute loss
        pred_actions = self.forward(image_seq, height_seq).mean
        loss = self.loss_fn(pred_actions, actions)

        # Step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
