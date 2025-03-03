import torch

state_dim = 10
action_dim = 8
num_episodes = 10000
update_frequency = 8
target_update_frequency = 200
lr = 0.001
epsilon = 0.8
epsilon_decay = 0.998
epsilon_min = 0.1
batch_size=64
gamma=0.95
buffer_size=1000000
save_interval=100

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.num_episodes = num_episodes
        self.save_interval = save_interval