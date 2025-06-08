"""
READ-ONLY: Runs behavior cloning and DAgger for homework 1
Hyperparameters for the experiment are defined in main()
"""

import os
import time
import argparse

from infrastructure.bc_trainer import BCTrainer
from agents.bc_agent import BCAgent
from agents.expert_agent import MPC
from config import Config


def run_bc(params):
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
        'full_policy': params['load_full'],
        'lstm_policy': params['load_lstm'],
        'cnn_policy': params['load_cnn'],
    }
    params['agent_class'] = BCAgent
    params['agent_params'] = agent_params

    #######################
    ## ENVIRONMENT PARAMS
    #######################

    params["env_name"] = config.env_config.env_name
    params["env_kwargs"] = {'render': False}

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy...')
    expert_agent = MPC(config)

    print('Done restoring expert policy...')

    ###################
    ### RUN TRAINING
    ###################

    trainer = BCTrainer(params)
    trainer.run_training_loop(
        n_iter=params['n_iter'],
        initial_expertdata=params['expert_data'],
        collect_policy=trainer.agent,
        eval_policy=trainer.agent,
        relabel_with_expert=params['do_dagger'],
        expert_policy=expert_agent,
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
    parser.add_argument('--do_dagger', action='store_true')

    # Sets the number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1500)     # Number of inner iterations
    parser.add_argument('--n_iter', '-n', type=int, default=1)                          # Number of dagger iterations

    # Load policy
    parser.add_argument('--load_cnn', type=str, default=None)
    parser.add_argument('--load_lstm', type=str, default=None)
    parser.add_argument('--load_full', type=str, default=None)

    # Amount of training data collected (in the env) during each iteration
    # To get a standard deviation, make sure batch size is N times the
    # number of steps per iteration, with N >> 1. It's fine to iterate with
    # default batch size, but for final results we recommend a batch size
    # of at least 10,000.
    # Amount of evaluation data collected (in the env) for logging metrics
    parser.add_argument('--eval_batch_size', type=int, default=2000)
    # Number of sampled data points to be used per gradient/train step
    parser.add_argument('--train_batch_size', type=int, default=500)

    # Replay buffer arguments
    # Collection
    parser.add_argument('--ep_len', type=int, default=150)                      # Max number of samples per episode
    parser.add_argument('--batch_size', type=int, default=10000)                # Number of steps during data collection
    parser.add_argument('--replay_buffer_max_ep', type=int, default=400)        # Max number of saved episodes
    # Sampling
    parser.add_argument('--replay_buffer_seq_len', type=int, default=25)        # Number of steps sampled in sequence

    # Learning rate for supervised learning
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--save_params', type=str)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Convert arguments to dictionary for easy reference
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'dagger_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) \
            of training, to iteratively query the expert and train \
            (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'bc_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data \
            just once (n_iter=1)')

    # Directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    run_bc(params)


if __name__ == "__main__":
    main()
