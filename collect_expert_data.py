import gymnasium as gym
from agents.expert_agent import MPC
from config import Config
import infrastructure.utils as utils
import pickle


config = Config()
params = {}
params['env_name'] = 'environment:environment/FlightArena'
params['env_kwargs'] = {'render': False}

env = gym.make(params['env_name'], **params['env_kwargs'])
expert_agent = MPC(config)

paths = utils.sample_n_trajectories(env, expert_agent, 600, max_path_length=200, expert=True, save_states=True)

for i in range(12):
    with open(f"agents/expert_data/mpc_{i+1}.pkl", "wb") as f:
        pickle.dump(paths[50*i:50*(i+1)], f)
