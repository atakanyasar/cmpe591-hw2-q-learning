import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from config import Config


class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device),  
            torch.LongTensor(np.array(actions, dtype=np.int64)).to(self.device),
            torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device),
            torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(self.device),  
            torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)



def init_target(target_model):
    global target
    target = target_model

def get_target():
    global target
    return target


class DQN(nn.Module):
    def __init__(self, config: Config):
        super(DQN, self).__init__()

        self.device = config.device
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim

        self.fc1 = nn.Linear(self.state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.action_dim)

        self.config = config
        self.replay_buffer = ReplayBuffer(config.buffer_size, self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
        self.loss = nn.SmoothL1Loss()


        self.__update_counter = 0
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state):
        if np.random.rand() < self.config.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state).argmax(dim=1).item()

    def update_target(self):
        get_target().load_state_dict(self.state_dict())

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size * 10:
            return

        self.__update_counter += 1
        if self.__update_counter % self.config.update_frequency != 0:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        q_values = self.forward(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = get_target()(next_states)
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values.max(dim=1, keepdim=True)[0] * (1 - dones.unsqueeze(1))

        loss = self.loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)  # Gradient Clipping
        self.optimizer.step()

        if self.__update_counter % self.config.target_update_frequency == 0:
            self.update_target()


    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


