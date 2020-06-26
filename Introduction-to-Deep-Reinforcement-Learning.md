# Introduction to Deep Reinforment Learning (DRL)
**Table of Contents**
- [Background: Machine Learning Method](#Background-Machine-Learning-Method)
- [Methodology](#Methodology)
    - [Reinforcement Learning](#Reinforcement-Learning)
        - [Q-Learning](#Q-Learning)
    - [Deep Q-Learning (DQL)](#Deep-Q-Learning-DQL)
- [Example: DQL Implementation](#Example-DQL-Implementation)
- [Comment](#Comment)
- [Reference](#Reference)

## Background: Machine Learning Method
There are three typical types of machine leanring methods:
1. Supervised Learning: given labeled data, train the model for predicting the correct result

<p>
<img src="./Image/Supervised Learning.png" alt="Supervised Learning" title="Supervised Learning" width="500">
</p>

2. Unsupervised Learning: given unlabeled data, train the model to find underlying patterns between data

<p>
<img src="./Image/Unsupervised Learning.png" alt="Unsupervised Learning" title="Unsupervised Learning" width="300">
</p>

3. Reinforcement Learning: get feedback (state and reward) from interacting with the environment, and adjust new action to the environment to get maximum expected reward. 

<p>
<img src="./Image/Reinforcement Learning.png" alt="Reinforcement Learning" title="Reinforcement Learning" width="500">
</p>

Method Comparison | Input Data | Output Result | Types of Problem | Application
:---:|:---:|:---:|:---:|:---:
Supervised Learning | Labeled data | Prediction result | Classification; Regression | Risk Evaluation; Forecasting
Unsupervised Learning | Unlabeled data | Underlying pattern | Clustering | Recommendation; Anomaly detection
Reinforcement Learning | Learn from environment | Action to the enviroment | Exploration and Exploitation | Self driving cars; Gaming

## Methodology
### Reinforcement Learning
As mensioned previously, Reinforcement Learning get feedback from interacting with the environment without having predefined data. It is a goal-oriented method that an agent tries to come up with the best action given a state. One of the most important issue in Reinforcement Learning is the design of reward function, which influence how fast the agent learns from interacting with the environment. 

For example, an utimate goal for a dog (agent) is to catch a frisbee thrown by a kid. The closer the dog to the frisbee, the more reward it will get. This reward function will affect the dog's subsequent action. The dog will know where it is (state) and how much reward it gets in the previous action. All these result will be saved as the dog's experience for deciding the next action.

#### Q-Learning
Q-learning is a model-free Reinforcement Learning algorithm. In Reinforcement Learning, agent will learn from experience. In Q-Learning, each state and action are viewed as inputs to a Q-function which outputs a corresponding Q-value (Expected future reward). Besides, these expereinces will be saved to a Q-table as a reference for agent to decide a best action.

<p>
<img src="./Image/Q-table Mechanism.png" alt="Q-table Mechanism" title="Q-table Mechanism" width="500">
</p>

### Deep Q-Learning (DQL)
In Q-Learning, the experience learned by the agent will be save to Q-table; however, when the problem scale is larger, Q-table will be ineffcient. Take playing games as an example, the action space and the state space are too large to handle. To deal with this problem, Neural Networks method is used to approximate the Q-value for each action when given a state. 

<p>
<img src="./Image/Deep Q Learning.png" alt="Deep Q Learning" title="Deep Q Learning" width="500">
</p>

## Example: DQL Implementation
**Environment**  
Open AI Gym: MountainCar-v0  
Description: The agent (a car) is started at the bottom of a valley. For any given state the agent may choose to accelerate to the left, right or cease any acceleration.

<p>
<img src="./Image/Mountain Car.PNG" alt="Mountain Car" title="Mountain Car" width="500">
</p>
  
Code shown below is adjust from [Reinforcement Learning 進階篇：Deep Q-Learning](https://medium.com/pyladies-taiwan/reinforcement-learning-%E9%80%B2%E9%9A%8E%E7%AF%87-deep-q-learning-26b10935a745)

Import Module: Pytorch is used for building neural network
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
```

Neural Network Structure:

```
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
```
Deep Q-Network module (simple version):
For more detail: [Deep Q-Learning](https://github.com/ITingHung/Introduction-to-Deep-Reinforcement-Learning/blob/master/Deep%20Q-learning.py)
```
class DQN(object):
    def __init__(self):
        # Create target network, evaluation network and memory
        
    def choose_action(self):
    	# Choose action according to the state

    def store_transition(self):
    	# Store experience to memory

    def learn(self):
    	# Update tatget network
```

Since the default reward setting is too simple, I revise it to accelerate the training process.  
- Default Reward  
Agent reached the flag (position = 0.5): 0  
Position of the agent is less than 0.5: -1  
- Adjust Reward
Distance between agent and the flag = pos-0.5 (negative)
Velocity of the agent = vel (positive if the agent is toward to the flag)
Reward = (pos-0.5)+vel
```
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
 ```
 
Result Plots:
- Default Reward
<p>
<img src="./Image/(Default Reward) Best Car Position in each Episode.png" alt="(Default Reward) Best Car Position in each Episode" title="(Default Reward) Best Car Position in each Episode" width="500">
</p>

Since there are only two values in reward space (-1; 0), the total reward in each episode is the same if the car didn't achieve the flag.

<p>
<img src="./Image/(Default Reward) Total Reward in each Episode.png" alt="(Default Reward) Total Reward in each Episode" title="(Default Reward) Total Reward in each Episode" width="500">
</p>

- Adjusted Reward
<p>
<img src="./Image/(Adjusted Reward) Best Car Position in each Episode.png" alt="(Adjusted Reward) Best Car Position in each Episode" title="(Adjusted Reward) Best Car Position in each Episode" width="500">
</p>
<p>
<img src="./Image/(Adjusted Reward) Total Reward in each Episode.png" alt="(Adjusted Reward) Total Reward in each Episode" title="(Adjusted Reward) Total Reward in each Episode" width="500">
</p>

Notice that the reward functions are different in these two cases, we can't compare the value of their Total Reward.

## Comment
1. According to the result in DQL Implementation, it is clear that the reward function has a great effect on the agent's action. A better designed reward function will lead to better effiency for an agent to learn from the environment.
2. In Deep Q-Learning, a neural network structure is used for dealing with large-scale problems. However, the model will become more complex which decrease its explanation. Besides, there are more hyper parameters needed to be tuned in the Deep Q-Network model.

## Reference
Machine Learning Method: [Supervised vs. Unsupervised vs. Reinforcement](https://www.aitude.com/supervised-vs-unsupervised-vs-reinforcement/)  
Reinforcement Learning: [Reinforcement Learning 健身房：OpenAI Gym](https://medium.com/pyladies-taiwan/reinforcement-learning-%E5%81%A5%E8%BA%AB%E6%88%BF-openai-gym-e2ad99311efc)  
Deep Q-Learning: [Reinforcement Learning 進階篇：Deep Q-Learning](https://medium.com/pyladies-taiwan/reinforcement-learning-%E9%80%B2%E9%9A%8E%E7%AF%87-deep-q-learning-26b10935a745)

