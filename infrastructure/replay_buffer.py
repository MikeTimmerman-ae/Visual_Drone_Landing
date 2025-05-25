"""
TO EDIT: A simple, generic replay buffer

Functions to edit:
    sample_random_data: line 103
"""
from infrastructure.utils import *


class ReplayBuffer():
    """
    Defines a replay buffer to store past trajectories

    Attributes
    ----------
    paths: list
        A list of rollouts
    obs: np.array
        An array of observations
    acs: np.array
        An array of actions
    rews: np.array
        An array of rewards
    next_obs:
        An array of next observations
    terminals:
        An array of terminals

    Methods
    -------
    add_rollouts:
        Add rollouts and processes them into their separate components
    sample_random_data:
        Selects a random batch of data
    sample_recent_data:
        Selects the most recent batch of data
    """
    def __init__(self, seq_len=25, max_ep=500):

        self.seq_len = seq_len
        self.max_ep = max_ep

        # Store each rollout
        self.paths = []
        self.traj_lengths = []

        # Store (concatenated) component arrays from each rollout
        self.ob_im = None
        self.ob_height = None
        self.acs = None
        self.rews = None
        self.next_ob_im = None
        self.next_ob_height = None
        self.terminals = None

    def __len__(self):
        if self.traj_lengths:
            return sum(self.traj_lengths)
        else:
            return 0

    def add_rollouts(self, paths):
        """
        Adds paths into the buffer and processes them into separate components

        :param paths: a list of paths to add
        :param concat_rew: whether rewards should be concatenated or appended
        """
        # Add new rollouts into our list of rollouts
        for path in paths:
            self.traj_lengths.append(len(path["reward"]))
            self.paths.append(path)

        # Convert new rollouts into their component arrays, and append them onto arrays
        obs_img, obs_height, actions, next_obs_img, next_obs_height, terminals, rewards, states = (
            convert_listofrollouts(paths))

        if self.ob_im is None:
            self.ob_im = obs_img
            self.ob_height = obs_height
            self.acs = actions
            self.rews = rewards
            self.next_ob_im = next_obs_img
            self.next_ob_height = next_obs_height
            self.terminals = terminals
            self.states = states
        else:
            self.ob_im = np.concatenate([self.ob_im, obs_img])
            self.ob_height = np.concatenate([self.ob_height, obs_height])
            self.acs = np.concatenate([self.acs, actions])
            self.rews = np.concatenate([self.rews, rewards])
            self.next_ob_im = np.concatenate([self.next_ob_im, next_obs_img])
            self.next_ob_height = np.concatenate([self.next_ob_height, next_obs_height])
            self.terminals = np.concatenate([self.terminals, terminals])
            self.states = np.concatenate([self.states, states])

        # Remove old episodes
        num_remove = max(0, len(self.traj_lengths) - self.max_ep)
        self.paths = self.paths[num_remove:]
        self.traj_lengths = self.traj_lengths[num_remove:]
        self.ob_im = self.ob_im[num_remove:]
        self.ob_height = self.ob_height[num_remove:]
        self.acs = self.acs[num_remove:]
        self.rews = self.rews[num_remove:]
        self.next_ob_im = self.next_ob_im[num_remove:]
        self.next_ob_height = self.next_ob_height[num_remove:]
        self.terminals = self.terminals[num_remove:]
        self.states = self.states[num_remove:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        """
        Samples a batch of random transitions

        :param batch_size: the number of transitions to sample
        :return:
            obs: a batch of observations
            acs: a batch of actions
            rews: a batch of rewards
            next_obs: a batch of next observations
            terminals: a batch of terminals
        """
        num_seq = batch_size // self.seq_len
        eligible_indices = [i for i, l in enumerate(self.traj_lengths) if l >= self.seq_len]
        selected_paths = np.random.choice(eligible_indices, size=num_seq, replace=False)

        sequences = {
            "obs_img": [],
            "obs_height": [],
            "acs": [],
            "rews": [],
            "next_obs_img": [],
            "next_obs_height": [],
            "terminals": [],
            "states": []
        }

        for idx in selected_paths:
            traj_len = self.traj_lengths[idx]
            traj_slice_start = np.random.randint(0, traj_len - self.seq_len + 1)
            traj_slice_end = traj_slice_start + self.seq_len

            sequences["obs_img"].append(self.ob_im[idx][traj_slice_start:traj_slice_end])
            sequences["obs_height"].append(self.ob_height[idx][traj_slice_start:traj_slice_end])
            sequences["acs"].append(self.acs[idx][traj_slice_start:traj_slice_end])
            sequences["rews"].append(self.rews[idx][traj_slice_start:traj_slice_end])
            sequences["next_obs_img"].append(self.next_ob_im[idx][traj_slice_start:traj_slice_end])
            sequences["next_obs_height"].append(self.next_ob_height[idx][traj_slice_start:traj_slice_end])
            sequences["terminals"].append(self.terminals[idx][traj_slice_start:traj_slice_end])
            sequences["states"].append(self.states[idx][traj_slice_start:traj_slice_end])

        for key in sequences:
            sequences[key] = np.stack(sequences[key])

        return sequences
