import gym
import random
import numpy as np

from DQN import *

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    space_size = [state_size, 64, action_size]
    agent = Agent(space_size)
    
    reward_history = []
    best_score = -np.inf
    file_path = 'NN_trained_params.pt'
    max_episodes = 10000;
    
    step_count = 0;
    for episode in range(max_episodes):
        state = env.reset()[0]
        #print(state)
        
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action = agent.select_action(state)
            state_next, reward, done, truncated, info = env.step(action)
            
            if done == True:
                done_val = 1
            else:
                done_val = 0
            
            episode_reward = episode_reward + reward;
            agent.replay_memory.push(state, action, reward, state_next, done_val)
            agent.learn()
            
            state = state_next
            agent.update_epsilon(step_count)
            step_count = step_count + 1;
            if step_count % agent.update_rate == 0:
                agent.update_target_model()
            
        reward_history.append(episode_reward)
        avg_score = np.mean(reward_history[-50:])
        print('Ep: {}, Total Steps: {}, Ep Score: {}, Avg Score: {}, Best: {}, Updated Espilon: {}'
              .format(episode, step_count, episode_reward, avg_score, np.round(best_score, 2),np.round(agent.epsilon, 10)))
        if avg_score >= best_score:
            agent.save_model(file_path)
            best_score = avg_score
            