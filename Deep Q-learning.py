# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:58:14 2020

@author: I-Ting Hung
@reference: https://medium.com/pyladies-taiwan/reinforcement-learning-%E9%80%B2%E9%9A%8E%E7%AF%87-deep-q-learning-26b10935a745
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # input: state to hidden layer, hidden layer to output: action
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        actions_value = self.out(x)
        return actions_value
    
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # size of experience in each memory: state + next state + reward + action
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # define for updating target network

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
    
        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # random
            action = np.random.randint(0, self.n_actions)
        else: # make the best choice according to policy
            actions_value = self.eval_net(x) # get the score from each action according to eval net
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # choose the action with highest score
    
        return action
    
    def store_transition(self, state, action, reward, next_state):
        # pack experience
        transition = np.hstack((state, [action, reward], next_state))
    
        # save into memory；old memory may be overlaped
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        # random sample batch_size experience
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])
    
        # calculate eval net and target net, get loss function of Q value
        q_eval = self.eval_net(b_state).gather(1, b_action) # calculate Q value according to eval net
        q_next = self.target_net(b_next_state).detach() # detach (target net won't be trained)
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # calculate Q value according to target net
        loss = self.loss_func(q_eval, q_target)
    
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # every target_replace_iter, update target net (copy eval net to target net)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

if __name__ == '__main__' :        
    env = gym.make('MountainCar-v0')
    
    # Environment parameters
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    
    # Hyper parameters
    n_hidden = 20
    batch_size = 32
    lr = 0.1                 # learning rate
    epsilon = 0.1             # epsilon-greedy
    gamma = 0.9               # reward discount factor
    target_replace_iter = 100 # target network update frequency
    memory_capacity = 2000
    n_episodes = 200
    
    # create DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)
    pos_his, reward_his = [], []
    
    # train DQN
    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        best_pos = -1.2 # min position defined in 'MountainCar-v0'
        state = env.reset()
        while True:
            env.render()
    
            # choose action
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            # revise reward to accelerate training process
            pos, vel = next_state
            r1 = pos-0.5 # better to make the car closer to the flag
            r2 = vel
            reward = r1+r2
            
            # save experience
            dqn.store_transition(state, action, reward, next_state)
            
            # record best position happened during steps
            best_pos = pos if (pos > best_pos) else best_pos
    
            # accumulate reward
            rewards += reward
    
            # train the model after having enough expereince
            if dqn.memory_counter > memory_capacity:
                dqn.learn()
    
            # go to next state
            state = next_state
    
            if done:
                pos_his.append(best_pos)
                reward_his.append(rewards)
                print(f'{i_episode+1} Episode finished after {t+1} timesteps, total rewards {rewards}')
                break
    
            t += 1
    
    env.close()
    
    plt.figure()
    plt.plot(pos_his)
    plt.xlabel('Episode')
    plt.ylabel('Best Car Position')
    plt.title('Best Car Position in each Episode')
    
    plt.figure()
    plt.plot(reward_his)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward in each Episode')