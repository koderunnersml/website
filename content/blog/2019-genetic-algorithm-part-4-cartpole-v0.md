---
title: 'Genetic Algorithm: Part 4 - CartPole-v0'
date: 2019-02-13T17:45:53.274Z
draft: true
categories: Podcast
tags:
  - MachineLearning
  - OptimizationTechniques
  - OpenAi
  - GeneticAlgorithm
author: KoderunnersML
authorImage: uploads/koderunners.jpg
comments: true
share: true
type: post
---
So far, we have learned the [basics](https://koderunners.ml/blog/2018-introduction-to-genetic-algorithm-part-2-implementation/) of Genetic Algorithm(GA) and solved a [classical problem using GA](https://koderunners.ml/blog/2019-genetic-algorithm-part-3-knapsack-problem/). GA can be applied to a variety of [real world problems](https://www.brainz.org/15-real-world-applications-genetic-algorithms/).

So, today we will use Open AI Gym environment to simulate a simple game known as CartPole-v0 and then we will use GA to automate the playing of the game. Sounds fun..... 

So, lets jump right into it.

# Problem Statement

A pole is standing upright on the cart. The goal is to balance the pole in an upright position by moving the cart left or right. You lose the game if the angle of the pole with the cart is more than 15 degrees. You win the game if you manage to keep the pole balanced for given number of frames. For every frame you mange to keep the pole in upright position you get a '+1' score.

# Solution

To solve this problem we will first create some random game data and then we will feed it to our GA model which in turn will predict the movement of cart(left or right) for every frame. For those of you new to GA, do refer to my previous tutorials.

As usual, we begin with importing libraries and making necessary initializations. 

**Code:**

```
import gym
import random
import numpy as np
from random import randint
from statistics import mean, median
from collections import Counter
```

```
env = gym.make("CartPole-v0")
env.reset()
#Number of frames
goal_steps = 500
score_requirement = 50
initial_games = 10000
```

If you are new to Open AI then you can check [this](https://medium.com/@ashish_fagna/understanding-openai-gym-25c79c06eccb) out. It will give you a better intuition of its terminologies.

Now we will collect the data by running the game environment 10000 times for radom moves i.e. for every frame it will be randomly decided whether our cart goes left or right. If our score is greater than or equal to 50 then we are going to store every move we made thereafter.

 Note that we are storing current action with previous observation. The reason behind it is that at first as the cart was still, there was no observation but when there was an action the observation was returned from the action. Hence, pairing the previous observation to the action we'll take.

We are considering '0' for 'left' and '1' for 'right'.

**Code:**

```
def create_data():
    training_data, scores, accepted_scores = [], [], []
    for _ in range(initial_games):
        score = 0
        #Moves from current environment and previous observations
        game_memory, prev_observation = [], []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
                
            prev_observation = observation
            score += reward
            
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append(data)
                    
        env.reset()        
        scores.append(score)
       
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    #print(Counter(accepted_scores))
        
    return training_data
```

We will follow the same flow chart for GA as we did earlier.

Initial population will be 8 randomly created set of genes/weights.

**Code:**
```
def create_initial_pop(pop_size):
    initial_pop = np.random.uniform(low = -2.0, high = 2.0, size = pop_size)
    print('Initial Population:\n{}'.format(initial_pop))
    return initial_pop
```
```
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
```
def predict(X):
    pred = np.empty((X.shape[0], 1))
    for i in range(X.shape[0]):
        if X[i] >= 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    return pred    
```
The fitness function that we will be using is :
