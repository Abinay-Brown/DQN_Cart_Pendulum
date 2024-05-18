'''
Reinforcement Learning Based Replay Memory Buffer
- Used to store {State, Action, Reward, State+1} at each step
- Random Minibatch sampling from Memory
'''
import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, total_capacity):
        self.total_capacity = total_capacity    # total Memory Capacity
        self.state_buffer = []                  # Initialize state buffer
        self.action_buffer = []                 # Initialize action buffer
        self.reward_buffer = []                 # Initialize reward buffer
        self.next_state_buffer = []             # Initialize next state buffer
        self.done_buffer = []                   # Initialize done buffer
        self.ind = 0

    def store(self, state, action, reward, next_state, done):

        if len(self.state_buffer) < self.total_capacity:
            self.state_buffer.append(state)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)
            self.next_state_buffer.append(next_state)
            self.done_buffer.append(done)
        else:
            self.state_buffer[self.ind] = state
            self.action_buffer[self.ind] = action
            self.reward_buffer[self.ind] = reward
            self.next_state_buffer[self.ind] = next_state
            self.done_buffer[self.ind] = done

        self.ind = (self.ind + 1) % self.total_capacity # for circular memory

    def sample(self, batch_size, device):
        sample_ind = random.sample(range(len(self.state_buffer)), batch_size)

        states      = torch.from_numpy(np.array(self.state_buffer)[sample_ind]).float().to(device)
        actions     = torch.from_numpy(np.array(self.action_buffer)[sample_ind]).to(device)
        rewards     = torch.from_numpy(np.array(self.reward_buffer)[sample_ind]).float().to(device)
        next_states = torch.from_numpy(np.array(self.next_state_buffer)[sample_ind]).float().to(device)
        dones       = torch.from_numpy(np.array(self.done_buffer)[sample_ind]).to(device)

        return states, actions, rewards, next_states, dones
        
    def mem_length(self):
        return len(self.buffer_state)