"""
Some miscellaneous utility functions

"""
import time
import numpy as np
from infrastructure.dynamics_utils import generate_init_states

############################################
############################################


def sample_trajectory(env, policy, max_path_length, init_state=None, expert=False, save_states=False):
    """
    Rolls out a policy and generates a trajectory

    :param policy: the policy to roll out
    :param max_path_length: the number of steps to roll out
    """
    # Initialize environment for the beginning of a new rollout
    policy.reset()
    ob, info = env.reset(options={'return_info': True, 'init_state': init_state})

    # Initialize data storage for across the trajectory
    obs_im, obs_height, acs, rewards, next_obs_im, next_obs_height, terminals, states = [], [], [], [], [], [], [], []
    while True:
        # Use the most recent observation to decide what to do
        obs_im.append(ob['cam'])
        obs_height.append(ob['radar'])
        if save_states:
            states.append(info['curr_state'])
        if expert:
            ac = policy.get_action(info['curr_time'], info['curr_state'], info['curr_action'])
        else:
            img = ob['cam']
            height = ob['radar']
            ac = policy.get_action(img, info['curr_state'])
        acs.append(ac)

        # Take that action and record results
        ob, rew, done, _, info = env.step(ac)

        # Record result of taking that action
        steps = info['curr_step']
        next_obs_im.append(ob['cam'])
        next_obs_height.append(ob['radar'])
        rewards.append(rew)

        # Rollout end due to done, or due to max_path_length
        rollout_done = 1 if done or steps >= max_path_length else 0
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs_im, obs_height, acs, rewards, next_obs_im, next_obs_height, terminals, states)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, expert=False, save_states=False):
    """
        Collect rollouts until we have collected `min_timesteps_per_batch` steps.
    """
    timesteps_this_batch = 0
    paths = []
    t0 = time.time()
    while timesteps_this_batch < min_timesteps_per_batch:
        # Collect rollout
        init_state = generate_init_states(1, method='uniform')[0]
        path = sample_trajectory(env, policy, max_path_length, init_state, expert=expert, save_states=save_states)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    print(f'[INFO] Finished generating {len(paths)} episodes in {time.time() - t0} sec. with {timesteps_this_batch} steps.')
    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, expert=False, save_states=False) -> list:
    """
        Collect `ntraj` rollouts.
    """
    paths = []
    t0 = time.time()
    init_states = generate_init_states(ntraj, 'sobol')
    fail_count = 0
    for i, init_state in enumerate(init_states):
        if (i+1) % 8 == 0:
            print(f"[INFO] Finished collection of {i+1} expert trajectories in {time.time() - t0} sec.")
        try:
            path = sample_trajectory(env, policy, max_path_length, init_state, expert=expert, save_states=save_states)
            paths.append(path)
        except Exception as e:
            fail_count += 1
            print(f"[WARNING] Trajectory {i + 1} failed at init_state={init_state} with error: {e}")
    print(f'[INFO] Finished generating {ntraj} episodes in {time.time() - t0} sec. with {fail_count} failed.')
    return paths

############################################
############################################

def Path(obs_img, obs_height, acs, rewards, next_obs_img, next_obs_height, terminals, states):
    """
        Take information (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"ob_image": np.array(obs_img, dtype=np.uint8),
            "ob_height": np.array(obs_height, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_obs_img": np.array(next_obs_img, dtype=np.uint8),
            "next_obs_height": np.array(next_obs_height, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "state": np.array(states, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    obs_img = np.array([path["ob_image"] for path in paths], dtype=object)
    obs_height = np.array([path["ob_height"] for path in paths], dtype=object)
    actions = np.array([path["action"] for path in paths], dtype=object)
    next_obs_img = np.array([path["next_obs_img"] for path in paths], dtype=object)
    next_obs_height = np.array([path["next_obs_height"] for path in paths], dtype=object)
    terminals = np.array([path["terminal"] for path in paths], dtype=object)
    rewards = np.array([path["reward"] for path in paths], dtype=object)
    states = np.array([path["state"] for path in paths], dtype=object)
    return obs_img, obs_height, actions, next_obs_img, next_obs_height, terminals, rewards, states

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
