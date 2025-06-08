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
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # → [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # → [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # → [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → [128, 1, 1]
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output: (x, y)
            nn.Tanh()
        )

        self.lstm = nn.LSTM(input_size=11, hidden_size=256, num_layers=1, batch_first=True)
        self.hidden_state = None
        self.prev_pos = None

        self.mean_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.ac_dim),  # 4 motor speeds
            nn.Tanh()
        )

        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Relu(),
            nn.Linear(128, 1)
        )

        self.to(ptu.device)

        # Parameter optimizer
        self.optimizer = optim.Adam(
            itertools.chain(
                self.cnn.parameters(),
                self.pos_head.parameters(),
                self.lstm.parameters(),
                self.mean_net.parameters(),
                self.value_head.parameters(),
            ),
            lr=self.learning_rate
        )
        self.lstm_optimizer = optim.Adam(
            itertools.chain(
                self.lstm.parameters(),
                self.mean_net.parameters(),
            ),
            lr=self.learning_rate
        )
        self.cnn_optimizer = optim.Adam(
            itertools.chain(
                self.cnn.parameters(),
                self.pos_head.parameters(),
            ),
            lr=self.learning_rate
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

    ##################################

    def save(self, filepath):
        """
        Save full policy weights
        """
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """
        Load full policy weights
        """
        self.load_state_dict(torch.load(filepath))

    def save_lstm(self, filepath):
        """
        Save LSTM + action head + logstd
        """
        torch.save({
            'lstm': self.lstm.state_dict(),
            'mean_net': self.mean_net.state_dict(),
            'logstd': self.logstd
        }, filepath)

    def load_lstm(self, filepath):
        """
        Load LSTM + action head + logstd
        """
        checkpoint = torch.load(filepath)
        self.lstm.load_state_dict(checkpoint['lstm'])
        self.mean_net.load_state_dict(checkpoint['mean_net'])
        self.logstd = checkpoint['logstd']

    def save_cnn(self, filepath):
        """
        Save CNN + pos head
        """
        torch.save({
            'cnn': self.cnn.state_dict(),
            'pos_head': self.pos_head.state_dict()
        }, filepath)

    def load_cnn(self, filepath):
        """
        Load CNN + pos head
        """
        checkpoint = torch.load(filepath)
        self.cnn.load_state_dict(checkpoint['cnn'])
        self.pos_head.load_state_dict(checkpoint['pos_head'])

    def reset(self):
        """ Reset hidden states """
        self.hidden_state = None
        self.prev_pos = None

    ##################################

    def forward(self, image_ob: torch.FloatTensor, state_ob: torch.FloatTensor, use_cnn=False) -> Any:
        """
        Defines the forward pass of the network

        :param image_seq:  (B, 3, 256, 256) image batch
        :param height_seq: (B, 1) scalar features
        :return:
            action: sampled action(s) from the policy
        """

        B = state_ob.shape[0]
        # Merge batch and time to encode images
        cnn_out = self.cnn(image_ob)                        # (B, 128)
        cnn_out = cnn_out.view(B, -1)
        cnn_out = F.layer_norm(cnn_out, cnn_out.shape[1:])

        # Get position from image embeddings
        pos_est = self.pos_head(cnn_out)
        if use_cnn:
            state_ob[:, 6:8] = pos_est

        # Construct state observations
        if self.prev_pos is None:
            pos_diff = torch.zeros((B, 3))
        else:
            pos_diff = state_ob[:, 6:9] - self.prev_pos
        self.prev_pos = state_ob[:, 6:9].detach()

        state_ob = torch.cat([
            state_ob[:, :2],                        # (pitch/roll)
            state_ob[:, 3:9],                       # (pitch/roll/yaw rate + xyz-pos)
            pos_diff                                # diff. in position
        ], dim=1)
        lstm_input = state_ob.unsqueeze(dim=1)      # (B, 1, 11)

        # Pass through LSTM
        if self.hidden_state is not None:
            h, c = self.hidden_state
            self.hidden_state = (h.detach(), c.detach())
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)  # (B, 1, 256)

        # Use last output to predict action
        mean = self.mean_net(lstm_out[:, -1])       # (B, 4)
        std = torch.exp(self.logstd)
        value = self.value_head(lstm_out[:, -1])    # (B, 1)
        return distributions.Normal(mean, std), value

    def update(self, image_ob: torch.FloatTensor, state_ob: torch.FloatTensor, actions: torch.FloatTensor, **kwargs):
        """
        Updates/trains the policy

        :param image_seq:  (B, 1, 256, 256) image batch
        :param height_seq: (B, 1) scalar features
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """

        # Compute loss
        pred_actions, _ = self.forward(image_ob, state_ob, kwargs['use_cnn']).mean
        loss = self.loss_fn(pred_actions, actions)

        # Step optimizer
        self.cnn_optimizer.zero_grad()
        loss.backward()
        self.cnn_optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_cnn(self, image_ob: torch.FloatTensor, state_label: torch.FloatTensor, eval=False):
        """
        Updates/trains the policy

        :param image_seq:  (B, 1, 256, 256) image batch
        :param state_label: (B, 12) scalar features
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        
        B = state_label.shape[0]
        # CNN forward pass
        cnn_out = self.cnn(image_ob)                        # (B, 128)
        cnn_out = cnn_out.view(B, -1)
        cnn_out = F.layer_norm(cnn_out, cnn_out.shape[1:])

        # Get position from image embeddings
        pos_est = self.pos_head(cnn_out)

        # Compute loss
        loss = self.loss_fn(pos_est, state_label[:, 6:8])

        if not eval:
            # Step optimizer
            self.cnn_optimizer.zero_grad()
            loss.backward()
            self.cnn_optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
