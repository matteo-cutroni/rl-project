import torch
from torch import nn


class Manager(nn.Module):
    def __init__(self, state_size, history_size, hidden_size=32):
        super(Manager, self).__init__()
        self.fc1 = nn.Linear(state_size + history_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3) # 0: act, 1:imagine from current state, 2: imagine from previously imagined state
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, state, history):
        x = torch.cat([state, history], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        route = self.softmax(x)
        return route
    
class Controller(nn.Module):
    def __init__(self, state_size, history_size, action_size, hidden_size=16):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(state_size + history_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, state, history):
        x = torch.cat([state, history], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action = self.softmax(x)
        return action
    
class Imagination(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Imagination, self).__init__()
        self.fc1 = nn.Linear(state_size + 1, hidden_size) # + 1 because I cat action
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_state = nn.Linear(hidden_size, state_size)
        self.fc_reward = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        next_state = self.fc_state(x)
        reward = self.fc_reward(x)
        return next_state, reward
    
class Memory(nn.Module):
    def __init__(self, input_size, history_size, num_layers=2):
        super(Memory, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size= history_size, num_layers=num_layers)

    def forward(self, d):
        output, _ = self.lstm(d)
        return output[-1, :]