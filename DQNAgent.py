import random
import numpy as np
import torch
import torch.nn.functional as F
from Model import Qnetwork
from ReplayMemory import ReplayMemory


class Agent:
    def __init__(self, state_dim, action_dim, device, eps_max, eps_min, eps_decay, total_capacity, discount=0.99, lr=1e-3):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.device = device

        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.buffer = ReplayMemory(total_capacity)
        self.online_net = Qnetwork(self.state_dim, 24, self.action_dim, lr).to(self.device)
        self.target_net = Qnetwork(self.state_dim, 24, self.action_dim, lr).to(self.device)
        self.target_net.eval()
            
        self.update_target()
        
    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_epsilon(self):   
        self.eps = max(self.eps_min, self.eps*self.eps_decay)


    def select_action(self, state):
        if random.random() <= self.eps:
            return random.sample(range(self.action_dim), 1)[0]

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action = torch.argmax(self.online_net(state))
        return action.item() 


    def learn(self, batchsize):
        # Get Random Minibacth of Transitions from Replay Memory
        states, actions, rewards, next_states, dones = self.buffer.sample(batchsize, self.device)
        
        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        
        q_pred = self.online_net(states).gather(1, actions.type(torch.int64)) 
        
        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        q_target = self.target_net(next_states).max(dim=1).values # because max returns data structure with values and indices
        q_target = q_target.reshape(-1, 1)
        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)
        
        # calculate the loss as the mean-squared error of yj and qpred
        loss = F.mse_loss(y_j, q_pred)
        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()
        

    def save_model(self, path):
        self.online_net.save_model(path)

    def load_model(self, path):
        
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.online_net.eval()