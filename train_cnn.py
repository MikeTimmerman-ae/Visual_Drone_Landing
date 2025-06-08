"""
READ-ONLY: Runs behavior cloning and DAgger for homework 1
Hyperparameters for the experiment are defined in main()
"""

import os
import time
import argparse

from infrastructure.cnn_trainer import CNNTrainer
from agents.bc_agent import BCAgent
from agents.expert_agent import MPC
from config import Config


def run_train(params):
    """
    Runs behavior cloning with the specified parameters

    Args:
        params: experiment parameters
    """

    #######################
    ## AGENT PARAMS
    #######################

    config = Config()
    params['config'] = config
    agent_params = {
        'learning_rate': params['learning_rate'],
        'replay_buffer_seq_len': params['replay_buffer_seq_len'],
        'replay_buffer_max_ep': params['replay_buffer_max_ep'],
    }
    params['agent_class'] = BCAgent
    params['agent_params'] = agent_params

    #######################
    ## ENVIRONMENT PARAMS
    #######################

    params["env_name"] = config.env_config.env_name
    params["env_kwargs"] = {'render': False}

    ###################
    ### RUN TRAINING
    ###################

    trainer = CNNTrainer(params)
    trainer.run_training_loop(
        expert_data=params['expert_data'],
    )


def main():
    """
    Parses arguments, creates logger, and runs behavior cloning
    """

    parser = argparse.ArgumentParser()
    # NOTE: The file path is relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--exp_name', '-exp', type=str,
        default='pick an experiment name', required=True)

    # Sets the number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--num_agent_train_steps', type=int, default=1001)     # Number of inner iterations

    # Amount of training data collected (in the env) during each iteration
    # To get a standard deviation, make sure batch size is N times the
    # number of steps per iteration, with N >> 1. It's fine to iterate with
    # default batch size, but for final results we recommend a batch size
    # of at least 10,000.
    # Number of sampled data points to be used per gradient/train step
    parser.add_argument('--train_batch_size', type=int, default=64)

    # Replay buffer arguments
    # Collection
    parser.add_argument('--replay_buffer_max_ep', type=int, default=400)        # Max number of saved episodes
    # Sampling
    parser.add_argument('--replay_buffer_seq_len', type=int, default=2)        # Number of steps sampled in sequence

    # Learning rate for supervised learning
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--scalar_log_freq', type=int, default=200)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Convert arguments to dictionary for easy reference
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    # Use this prefix when submitting. The auto-grader uses this prefix.
    logdir_prefix = 'cnn_'

    # Directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    run_train(params)


if __name__ == "__main__":
    main()
