# Introduction to Deep Reinforment Learning (DRL)
**Table of Contents**
- [Machine Learning Method](#Machine-Learning-Method)
- [Reinforcement Learning](#Reinforcement-Learning)
- [Deep Reinforcement Learning](#Deep-Reinforcement-Learning-DRL)
- [DRL Implementation](#DRL-Implementation)
- [Reference](#Reference)

## Machine Learning Method
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

## Reinforcement Learning
As mensioned previously, Reinforcement Learning get feedback from interacting with the environment without having predefined data. It is a goal-oriented method that an agent tries to come up with the best action given a state. One of the most important issue in Reinforcement Learning is the design of reward function, which influence how fast the agent will learn from the experience of interacting with the environment. 

For example, an utimate goal for a dog (agent) is to catch a frisbee thrown by a kid. The closer the dog to the frisbee, the more reward it will get. This reward function will affect the dog's subsequent action. The dog will know where it is (state) and how much reward it gets in the previous action. All these result will be saved as the dog's experience for deciding the next action.

The experience learned from each state and action will be viewed as inputs to a Q-function and output a Q-value (Expected future reward). Besides, these expereinces will be saved to a Q-table as a reference for agent to decide a best action.

<p>
<img src="./Image/Q-table Mechanism.png" alt="Q-table Mechanism" title="Q-table Mechanism" width="500">
</p>

## Deep Reinforcement Learning (DRL)
In Reinforcement Learning, the experience learned by the agent will be save to Q-table; however, when the action space or the state space is too large (for example: player games), Q-table will be ineffcient. To deal with this problem, Neural Networks method is used to approximate the Q-value for each action when given a state. 

<p>
<img src="./Image/Deep Q Learning.png" alt="Deep Q Learning" title="Deep Q Learning" width="500">
</p>

## DRL Implementation

## Reference
Machine Learning Method: [Supervised vs. Unsupervised vs. Reinforcement](https://www.aitude.com/supervised-vs-unsupervised-vs-reinforcement/) 
Reinforcement Learning: [Reinforcement Learning 健身房：OpenAI Gym](https://medium.com/pyladies-taiwan/reinforcement-learning-%E5%81%A5%E8%BA%AB%E6%88%BF-openai-gym-e2ad99311efc)
Deep Reinforcement Learning: [Reinforcement Learning 進階篇：Deep Q-Learning](https://medium.com/pyladies-taiwan/reinforcement-learning-%E9%80%B2%E9%9A%8E%E7%AF%87-deep-q-learning-26b10935a745)

