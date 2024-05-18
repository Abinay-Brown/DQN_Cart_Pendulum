'''
Reinforcement Learning
Implementation of the Qnetwork that takes continous states and returns
probablility of discrete actions to take
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Qnetwork(nn.Module):

    def __init__(self, state_shape, hidden_shape, action_shape, lr=1e-3):
        super(Qnetwork, self).__init__()
        
        self.dense1 = nn.Linear(in_features=state_shape, out_features = hidden_shape)
        self.dense2 = nn.Linear(in_features=hidden_shape, out_features = hidden_shape)
        self.dense3 = nn.Linear(in_features=hidden_shape, out_features = action_shape)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))