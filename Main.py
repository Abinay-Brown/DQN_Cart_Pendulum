import torch
import numpy as np
from DQNAgent import *
from TrainTest import *
from Environment import *

        
if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_mode = True
    Path = 'saved_mode.pt'
    # Initialize the Environment
    initial_conditions = np.array([0, np.pi, 0, 0]); # pos (m), angle (rad), vel (m/s), angular_vel (rad/s)
    action_space = 3*np.array([-2.0, 2.0]) # Action Space (Newtons)
    episode_length = 500; # Epsiode Length (steps)
    settling_length = 30;  # Settling Time Length (steps)
    params = [0.2, 0.5, 0.3, 0.001, 0]; # m, M, L, d, u
    termination = 30; # seconds
    env = Environment(initial_conditions, action_space, episode_length, settling_length, params, termination);
    
    output_path = 'data.pt'
    
    if training_mode == True:
        
        state_dim = len(initial_conditions)
        action_dim = len(action_space)
        eps_max = 1
        eps_min = 0.01
        eps_decay = 0.9999
        total_capacity = 256
        discount = 0.99
        lr = 1e-4
        DQNagent = Agent(state_dim, action_dim, device, eps_max, eps_min, eps_decay, total_capacity, discount, lr)
        
        total_episodes = 10000
        memory_episodes = 256
        batchsize = 64
        update_freq = 10
        
        train(env, DQNagent, total_episodes, memory_episodes, batchsize, update_freq, output_path)
    else:
        state_dim = len(initial_conditions)
        action_dim = len(action_space)
        eps_max = 0
        eps_min = 0
        eps_decay = 0
        total_capacity = 0
        discount = 0
        lr = 0
        DQNagent = Agent(state_dim, action_dim, device, eps_max, eps_min, eps_decay, total_capacity, discount, lr)
        DQNagent.load_model(output_path)
        test(env, DQNagent, 1)
             
 