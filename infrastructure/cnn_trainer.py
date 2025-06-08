"""
Defines a trainer which updates a behavior cloning agent

"""

from collections import OrderedDict

import pickle
import time
import torch
import gymnasium as gym
import numpy as np

from infrastructure import pytorch_util as ptu
from infrastructure.logger import Logger
from infrastructure import utils
from tqdm import trange, tqdm


class CNNTrainer:
    """
    A class which defines the training algorithm for the agent. Handles
    sampling data, updating the agent, and logging the results.

    ...

    Attributes
    ----------
    agent : BCAgent
        The agent we want to train

    Methods
    -------
    run_training_loop:
        Main training loop for the agent
    collect_training_trajectories:
        Collect data to be used for training
    train_agent
        Samples a batch and updates the agent
    """

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get parameters, create logger, and create the TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Set logger attributes
        self.log_metrics = True

        #############
        ## ENV
        #############

        # Make the gym environment
        self.params['env_kwargs']['render_mode'] = None
        self.env = gym.make(self.params['env_name'], **self.params['env_kwargs'])
        self.env.reset(seed=seed)

        # Set observation and action sizes
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim

        # Define the simulation timestep, which will be used for video saving
        self.fps = self.env.metadata['render_fps']

        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])

    def run_training_loop(self, expert_data=None):
        """ Samples a batch of trajectories and updates the agent with the batch
        :param expert_data:
        """       

        # Collect trajectories, to be used for training
        paths, envsteps = self.collect_training_trajectories(expert_data)
        self.envsteps = envsteps
        n_traj =  len(paths)

        np.random.shuffle(paths)
        split_idx = int(0.9 * n_traj)
        train_paths = paths[:split_idx]
        eval_paths = paths[split_idx:]

        # Add collected data to replay buffer
        self.agent.add_to_replay_buffer(paths)

        # Initialize variables at beginning of training
        all_logs = []
        self.start_time = time.time()

        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in trange(self.params['num_agent_train_steps'], desc="Training Steps"):
            
            # Log and save videos and metrics
            if self.log_metrics:
                if self.params['save_params']:
                    tqdm.write('\nSaving agent params')
                    self.agent.save_cnn('{}/cnn_itr_{}.pt'.format(self.params['logdir'], train_step))
                # Perform logging
                tqdm.write('Beginning logging procedure...')
                self.perform_logging(train_step, train_paths, eval_paths, all_logs)

            # Decide if metrics should be logged
            if (train_step+1) % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # Train agent (using sampled data from replay buffer)
            batch_size = self.params['train_batch_size']
            sequences = self.agent.sample(batch_size)

            train_log = self.agent.train_cnn(sequences['obs_img'], sequences['states'])
            all_logs.append(train_log)
            tqdm.write(f"Current training step {train_step + 1} with loss {train_log['Training Loss']:.4f}")

    ####################################
    ####################################

    def collect_training_trajectories(self, expert_data):
        """
        :param expert_data: path to expert data pkl file
        :return:
            paths: a list trajectories
            envsteps: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # Load training data or use the current policy to collect more data
        print("\nCollecting data to be used for training...")
        if expert_data:
            print(f"Loading expert data from {expert_data}...")
            paths = []
            for i in range(6):
                with open(expert_data + f"_{i+1}.pkl", 'rb') as f:
                    paths += pickle.load(f)
            envsteps = sum([utils.get_pathlength(path) for path in paths])
        
        print(f"\nUsing {envsteps} collected datasamples...")
        return paths, envsteps

    ####################################
    ####################################

    def perform_logging(self, train_step, train_paths, eval_paths, training_logs):
        """
        Logs training trajectories and evals the provided policy to log
        evaluation trajectories and videos

        :param train_step:
        :param paths: paths collected during training that we want to log
        :param eval_policy: policy to generate eval logs and videos
        :param training_logs: additional logs generated during training
        """
        
        # Save evaluation metrics
        if self.log_metrics:

            # Get the returns and episode lengths of all paths, for logging
            train_returns = []
            for i, path in enumerate(tqdm(train_paths, desc="Evaluating Train Paths")):
                loss = self.agent.eval_cnn(path["ob_image"], path["state"])
                train_returns.append(loss)
                if i % 10 == 0:
                    tqdm.write(f"Iter {train_step} - Train Loss: {loss:.4f}")

            eval_returns = []
            for i, path in enumerate(tqdm(eval_paths, desc="Evaluating Eval Paths")):
                loss = self.agent.eval_cnn(path["ob_image"], path["state"])
                eval_returns.append(loss)
                if i % 1 == 0:
                    tqdm.write(f"Iter {train_step} - Eval Loss: {loss:.4f}")

            # Define logged metrics
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)

            logs["Train_EnvstepsSoFar"] = self.envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time


            # Perform the logging with tensorboard
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, train_step)
            # Log enire loss over epochs
            for i, value in enumerate(training_logs):
                self.logger.log_scalar(value["Training Loss"], "Train Loss", i + train_step * self.params['scalar_log_freq'] + 1)
            print('Done logging...\n\n')

            self.logger.flush()