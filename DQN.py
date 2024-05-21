import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

class ReplayMemory:
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen =  max_capacity)
    
    def push(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))
    
    def sample(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        return minibatch
    def getCapacity(self):
        return len(self.memory)

class Qnetwork(nn.Module):

    def __init__(self, state_size, hidden_size, action_size):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(in_features = state_size, out_features = hidden_size)
        self.fc2 = nn.Linear(in_features = hidden_size, out_features = hidden_size)
        self.fc3 = nn.Linear(in_features = hidden_size, out_features = action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        
class Agent:
    def __init__(self, space_size):
        state_size, hidden_size, action_size = space_size[0:]; 
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 100000
        self.learning_rate = 0.001
        self.update_rate = 1000
        self.max_capacity = 100000
        self.minibatch_size = 32    
        
        # Q network and Replay Memory        
        self.replay_memory = ReplayMemory(self.max_capacity)
        self.policy_net = Qnetwork(self.state_size, self.hidden_size, self.action_size)
        self.target_net = Qnetwork(self.state_size, self.hidden_size, self.action_size)
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # MSE loss function and Adam optimizer
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learning_rate)
    
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self, step_count):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)*np.exp(-1*step_count/self.epsilon_decay)
        
    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(state)
        
        return torch.argmax(action).item()
    
    def learn(self):
        if self.replay_memory.getCapacity() <= self.minibatch_size:
            return
        
        minibatch = self.replay_memory.sample(self.minibatch_size)
        states_list = []
        actions_list = []
        rewards_list = []
        states_next_list = []
        dones_list = []
        current_q_list = []
        target_q_list = []
        
        for state, action, reward, state_next, done in minibatch:
            if done:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    state_next = torch.FloatTensor(state_next).unsqueeze(0)
                    target = torch.FloatTensor(reward + self.gamma*self.target_net(state_next).max())
            state = torch.FloatTensor(state).unsqueeze(0)
            current_q = self.policy_net(state)
            current_q_list.append(current_q)
            
            target_q = self.target_net(state)
            
            target_q[0, action] = target
            target_q_list.append(target_q)
            
        loss = self.loss_function(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''            
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            states_next_list.append(state)
            dones_list.append(done)
        '''
            
        '''  
        states = torch.FloatTensor(states_list)
        actions = torch.LongTensor(actions_list).unsqueeze(1)
        rewards = torch.FloatTensor(rewards_list)
        states_next = torch.FloatTensor(states_next_list)
        dones = torch.FloatTensor(dones_list)
        
        q_pred = self.policy_net(states).gather(1, actions)
        
        q_target = self.target_net(states_next).max(1).values
        q_target = q_target.reshape(-1, 1)
        
        q_target[dones[:] == 1] = 0
        y_j = rewards + (self.gamma * q_target)
        loss = self.loss_function(y_j, q_pred)
        self.optimizer.zero_grad()
        loss.backward()
        '''
            
        
    
    def save_model(self, path):
        self.policy_net.save_model(path)

    def load_model(self, path):
        
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        
        