import torch 
from DQNAgent import *

def FillBuffer(env, DQNagent, memory_episodes):

    for eps_count in range(memory_episodes):
        done = False
        env.state = env.reset()

        while not done:
            action = env.action_sample()
            next_state, reward, done = env.step(action)
            DQNagent.buffer.store(env.state, action, reward, next_state, done)
            env.state = next_state
            
        print([eps_count, memory_episodes])


def train(env, agent, total_epsiodes, memory_episodes, batchsize, update_freq, model_filename):
    FillBuffer(env, agent, memory_episodes);
    
    step_count = 0
    reward_history = []
    best_score = -np.inf
    
    for ep_count in range(total_epsiodes):
        state = env.reset()
        done = False;
        ep_reward = 0;
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.store(state, action, reward, next_state, done)
            agent.learn(batchsize)
            
            if np.mod(step_count,  update_freq) == 0:
                agent.update_target()
            env.state = next_state
            ep_reward += reward
            step_count += 1
        print([env.step_invert])
        agent.update_epsilon()
        reward_history.append(ep_reward)
        current_avg_score = np.mean(reward_history[-100:])
        print('Ep: {}, Total Steps: {}, Ep Score: {}, Avg Score: {}, Updated Espilon: {}'.format(ep_count, step_count, ep_reward, current_avg_score, agent.eps))
        
        if current_avg_score >= best_score:
            agent.save_model(model_filename)
            best_score = current_avg_score
    
    
def test(env, agent, test_eps):
    for ep_count in range(test_eps):
        state = env.reset();
        done = False
        ep_reward = 0
        while not done:
            
            action = agent.select_action(state)
            next_state, reward, done = env.step(action);
            
            env.state = next_state
            ep_reward += reward
            print([env.state[0], np.mod(env.state[1], 2*np.pi)])
