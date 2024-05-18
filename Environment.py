import numpy as np
import random
from numpy import pi, sin, cos, mod
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def cart_pend_dynamics(state, t, m, M, L, d, u):
    x, theta, v, w = state[0:4];
    cx = cos(theta)
    sx = sin(theta)
    g = 9.81
    den = m * (L**2) * (M + m*(1 - cx**2)); 
    xddot = ((-(m**2)*(L**2)*g*cx*sx) + (m*(L**2)*(m*L*(w**2)*sx -  d*v)) + (m*(L**2)*u))/den
    thetaddot = ((((m + M)*m*g*L*sx)) - (m*L*cx*(m*L*(w**2)*sx - d*v)) + (m*L*cx*u))/den;
    statedot = [v, w, xddot, thetaddot];
    return statedot   

def plot_epsiode(episode, tspan):
    
    x = epsiode[:, 0]
    theta = epsiode[:, 1]
    v = epsiode[:, 2]
    w = epsiode[:, 3]
    u = epsiode[:, 4]
    
    plt.subplot(2, 2, 1)
    plt.plot(tspan, x)
    plt.xlabel(" Time (sec)")
    plt.ylabel(" Displacement (m)")
    
    plt.subplot(2, 2, 2)
    plt.plot(tspan, v)
    plt.xlabel(" Time (sec)")
    plt.ylabel(" Velocity (m/s)")
    
    plt.subplot(2, 2, 3)
    plt.plot(tspan, theta)
    plt.xlabel(" Time (sec)")
    plt.ylabel(" Theta (rad)")
    
    plt.subplot(2, 2, 4)
    plt.plot(tspan, theta_dot)
    plt.xlabel(" Time (sec)")
    plt.ylabel(" Angular Velocity (rad/s)")
    plt.show()
    
    
    
    
    
class Environment:
    def __init__(self, IC, AS, EL, ST, params, term):
        '''
        IC: Initial Conditions
        AS: Action Space
        EL: Epsiode Length (steps)
        ST: Settling Time (steps)
        params: [m, M, L, d, u];
        term: Total Simulation time (sec)
        '''
        self.IC = IC
        self.state = IC;         
        self.action_space = AS;
        self.init_cond = IC;
        self.step_invert = 0;
        self.step_counter = 0;
        self.episode_duration = EL;
        self.ST = ST
        self.dt = 0.05;
        self.params = params;
    
    def action_sample(self):
        return random.sample(range(len(self.action_space)), 1)[0]
                   
    def reset(self):
        self.state = self.IC;
        self.step_invert = 0;
        self.step_counter = 0;
        
        return self.state
    
    def step(self, action):
        
        u = self.action_space[action];
        self.params[-1] = u;
        tstep = np.linspace(0, self.dt, 2);
        m, M, L, d, u = self.params
        sol = odeint(cart_pend_dynamics, self.state, tstep, args=(m, M, L, d, u))   
        next_state = sol[-1, :]
        
        angle_tol = (pi/180)*12
        
        # if pole is upright
        if np.abs(mod(sol[-1, 1], 2*pi) - pi) <= angle_tol:
            reward = 1;
            # if pole is previously upright and currently upright then increment
            #if np.abs(mod(self.state[1], 2*pi) - pi) <= angle_tol and np.abs(mod(next_state[1], 2*pi) - pi) < angle_tol: 
            #    self.step_invert = self.step_invert + 1
                
            #else:
                # Otherwise reset the inverted step counter
            #    self.step_invert = 0;
        else:
            reward = 0;
            
        if self.step_counter == self.episode_duration - 1:
            done = True
            #print('Flag 2')
        elif -10 > sol[-1, 0] or 10 < sol[-1, 0]:
            done = True
            #print('Flag 3') 
        else:
            done = False
        
        self.step_counter = self.step_counter + 1;
        return next_state, reward, done 
    