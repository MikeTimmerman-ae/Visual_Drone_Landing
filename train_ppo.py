from stable_baselines3.ppo import PPO
import gymnasium as gym
from agents.bc_agent import BCAgent
from stable_baselines3.common.vec_env import DummyVecEnv

params = {}
params["env_kwargs"] = {'render': False}
params['agent_params'] = agent_params = {
    'learning_rate': 5e-3,
    'replay_buffer_seq_len': 25,
    'replay_buffer_max_ep': 750,
    'ac_dim': 4,
    'full_policy': None,
    'lstm_policy': 'data/dagger_lstm_05-06-2025_18-06-50/lstm_itr_2.pt',
    'cnn_policy': 'data/cnn_final_06-06-2025_17-59-53/cnn_itr_1000.pt',
}
agent = BCAgent(params['agent_params'])
env = DummyVecEnv([lambda: gym.make('environment:environment/FlightArena', **params['env_kwargs'])])





