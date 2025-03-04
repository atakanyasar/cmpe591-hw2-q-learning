import matplotlib.pyplot as plt
import numpy as np
import time
import os
from homework2 import Hw2Env
from dqn import DQN
from config import Config
from dqn import get_target, init_target
import csv
import torch

def test():
    config = Config()
    dqn = DQN(config)
    dqn.load_state_dict(torch.load("dqn_model.pth"))
    dqn.eval()

    env = Hw2Env(n_actions=config.action_dim, render_mode="gui")

    for episode in range(config.num_episodes):
        env.reset()
        state = env.high_level_state()
        prev_state = state[:4]
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        start_time = time.time()

        while not done:

            concatenated_state = np.concatenate((state, prev_state), axis=None)
            _state = torch.FloatTensor(concatenated_state).to(config.device).unsqueeze(0)
            with torch.no_grad():
                action = dqn.forward(_state).argmax(dim=1).item()

            next_state, reward, is_terminal, is_truncated = env.step(action)
    
            done = is_terminal or is_truncated
            cumulative_reward += reward
            episode_steps += 1
            concatenated_next_state = np.concatenate((next_state, state[:4]), axis=None)

            prev_state = state[:4]
            state = next_state
        
        end_time = time.time()


if __name__ == "__main__":
    test()