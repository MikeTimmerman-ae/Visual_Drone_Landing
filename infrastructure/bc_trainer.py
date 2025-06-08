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

# The number of rollouts to save to videos in PyTorch
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # Constant for video length, we overwrite this in the code below


class BCTrainer:
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
    do_relabel_with_expert
        Relabels trajectories with new actions for DAgger
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
        self.log_video = True
        self.log_metrics = True

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params['video_log_freq'] == -1:
            self.params['env_kwargs']['render_mode'] = None
        self.env = gym.make(self.params['env_name'], **self.params['env_kwargs'])
        self.env.reset(seed=seed)

        # Set the maximum length for episodes and videos
        MAX_VIDEO_LEN = self.params['ep_len']

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

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # Initialize variables at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # Decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # Collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy
            )
            paths, envsteps_this_batch = training_returns

            # Relabel the collected observations with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths, envsteps_this_batch = self.do_relabel_with_expert(expert_policy, paths)

            self.total_envsteps += envsteps_this_batch

            # Add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # Train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()

            # Log and save videos and metrics
            if self.log_video or self.log_metrics:

                if self.params['save_params'] == "lstm":
                    print('\nSaving LSTM agent params')
                    self.agent.save_lstm('{}/lstm_itr_{}.pt'.format(self.params['logdir'], itr))
                elif self.params['save_params'] == "cnn":
                    print('\nSaving CNN agent params')
                    self.agent.save_cnn('{}/cnn_itr_{}.pt'.format(self.params['logdir'], itr))
                elif self.params['save_params'] == "full":
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

                # Perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, training_logs)

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy
    ):
        """
        :param itr:
        :param load_initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # Load training data or use the current policy to collect more data
        print("\nCollecting data to be used for training...")
        if itr == 0 and load_initial_expertdata:
            print(f"Loading expert data from {load_initial_expertdata}...")
            paths = []
            for i in range(6):
                with open(load_initial_expertdata + f"_{i+1}.pkl", 'rb') as f:
                    paths += pickle.load(f)
            envsteps_this_batch = sum([utils.get_pathlength(path) for path in paths])
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, self.params['batch_size'],
                                                                   self.params['ep_len'], False, True)
        print(f"\nUsing {envsteps_this_batch} collected datasamples...")
        return paths, envsteps_this_batch

    def train_agent(self):
        """
        Samples a batch of trajectories and updates the agent with the batch
        """
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # Sample data from the data replay buffer
            batch_size = self.params['train_batch_size']
            sequences = self.agent.sample(batch_size)

            # Use the sampled data to train an agent
            train_log = self.agent.train(sequences['obs_img'], sequences['states'], sequences['acs'], True)
            print(f'Current training step {train_step + 1} with loss {train_log["Training Loss"]}')
            all_logs.append(train_log)
        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        """
        Relabels collected trajectories with an expert policy

        :param expert_policy: the policy we want to relabel the paths with
        :param paths: paths to relabel
        """
        print("\nRelabelling collected observations with labels from an expert policy...")

        # Relabel collected obsevations (from our policy) with labels from an expert policy
        paths_out = []
        action = np.zeros((4, ))
        for path in paths:
            states = path["state"]
            for step, state in enumerate(states):
                try:
                    action = expert_policy.get_action(0, state, action)
                    path["action"][step] = action
                except Exception as e:
                    print(f"[WARNING] Failed to generate action at {step + 1} steps with error: {e}")
                    break
            if step >= self.params['agent_params']['replay_buffer_seq_len']:
                path["ob_image"] = path["ob_image"][:step]
                path["ob_height"] = path["ob_height"][:step]
                path["reward"] = path["reward"][:step]
                path["action"] = path["action"][:step]
                path["next_obs_img"] = path["next_obs_img"][:step]
                path["next_obs_height"] = path["next_obs_height"][:step]
                path["terminal"] = path["terminal"][:step]
                path["state"] = path["state"][:step]
                paths_out.append(path)
        envsteps_this_batch = sum([utils.get_pathlength(path) for path in paths_out])
        print(f"[INFO] Added {len(paths_out)} relabelled paths to the replay buffer with {envsteps_this_batch} steps.")
        return paths_out, envsteps_this_batch

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, training_logs):
        """
        Logs training trajectories and evals the provided policy to log
        evaluation trajectories and videos

        :param itr:
        :param paths: paths collected during training that we want to log
        :param eval_policy: policy to generate eval logs and videos
        :param training_logs: additional logs generated during training
        """

        # Save evaluation rollouts as videos in tensorboard event file
        if self.log_video:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN)

            # Save training and evaluation videos
            print('\nSaving rollouts as videos...')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, 
                                            max_videos_to_save=MAX_NVIDEO, 
                                            video_title='eval_rollouts')

        # Save evaluation metrics
        if self.log_metrics:
            # Collect evaluation trajectories, for logging
            print("\nCollecting data for eval...")
            eval_paths, _ = utils.sample_trajectories(
                self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'], False, True
            )
            with open(f"agents/expert_data/eval_policy_{itr}.pkl", "wb") as f:
                pickle.dump(eval_paths, f)

            # Get the returns and episode lengths of all paths, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # Define logged metrics
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now from additional training logs
            logs.update(last_log)


            if itr == 0:
                self.initial_return = np.mean(train_returns)
                logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # Perform the logging with tensorboard
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            # Log enire loss over epochs
            for i, value in enumerate(training_logs):
                self.logger.log_scalar(value["Training Loss"], "Train Loss", i + itr * self.params['num_agent_train_steps_per_iter'] + 1)
            print('Done logging...\n\n')

            self.logger.flush()