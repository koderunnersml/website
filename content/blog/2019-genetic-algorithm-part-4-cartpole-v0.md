---
title: 'Genetic Algorithm: Part 4 - CartPole-v0'
date: 2019-02-13T18:09:45.011Z
draft: false
categories: Podcast
tags:
  - MachineLearning
  - OptimizationTechniques
  - OpenAI
  - GeneticAlgorithm
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/g4.jpeg
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
import numpy as npimport matplotlib.pyplot as plt
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

![](/uploads/fitness_function.gif)

where,

\    w = weights

\    x = observations

**Code:**

```
def cal_fitness(population, X, y, pop_size):
    fitness = np.empty((pop_size[0], 1))
    for i in range(pop_size[0]):
        hx  = X@(population[i]).T
        fitness[i][0] = np.sum(hx)
    return fitness
```

We will select the fittest individuals as paretns i.e. solutions with the highest fitness value.

**Code:**

```
def selection(population, fitness, num_parents):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents
```

Selected parents will be mated to produce offsprings.

**Code:**

```
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = random.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings    
```

Now we will mutate few offsprings to create diversity in solutions and for that we will add some noise to randomly selected gene of individuals selected for mutation.

**Code:**

```
def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = random.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        mutants[i,int_random_value] += np.random.uniform(-1.0, 1.0, 1)
        
    return mutants
```

Now we will call the functions defined above in the order of flow chart.

**Code:**

```
def GA_model(training_data):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)
    
    weights = []
    num_solutions = 8
    pop_size = (num_solutions, X.shape[1])
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    num_generations = 50
    
    population = create_initial_pop(pop_size)
    
    for i in range(num_generations):
        fitness = cal_fitness(population, X, y, pop_size)
        parents = selection(population, fitness, num_parents)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
    
    fitness_last_gen = cal_fitness(population, X, y, pop_size)
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    weights.append(population[max_fitness[0][0],:])
    return weights
```

```
def GA_model_predict(test_data, weights):
    hx = sigmoid(test_data@(weights).T)
    pred = predict(hx)
    pred = pred.astype(int)
    return pred[0][0]
```

```
training_data = create_data()
weights = GA_model(training_data)
print('Weights: {}'.format(weights))
weights = np.asarray(weights)
```

**Output:**

```
Initial Population:
[[ 0.67999273  1.20045524 -0.31810563 -1.14804361]
 [-1.51475165 -1.42250336  0.03428274  0.63371852]
 [-0.8970108   1.03936397  1.84329259  1.72682724]
 [ 1.94204407 -0.77717282  0.14019162 -1.1903907 ]
 [ 0.41835458 -1.22852332  0.9296547  -1.12009693]
 [-0.65292285 -1.40827788  1.55964313 -0.23029554]
 [-1.44485637  0.02821767  0.48371509 -0.67509993]
 [ 1.51550571 -1.66566025  0.17737747 -1.76249427]]
Weights: [array([ 7.53967288, 14.12987549, -2.29013767, -8.39335431])]
```

Lets visualize the fitness growth with respect to generations.

**Code:**

```
fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
```

**Output:**

![](/uploads/openai1.png)

So everything is set and good to go. We will feed our model with training data and then run the game environment 10 times. Initially, we will decide a random move and based on that our model will play the game.

**Code:**

```
scores, choices = [], []
for each_game in range(10):
    score = 0
    game_memory, prev_obs = [], []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = GA_model_predict(prev_obs, weights)
        choices.append(action)    
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)        
print('Required Score:',str(score_requirement))    
print('Average Score Achieved:',sum(scores)/len(scores))env.close()
```

**Output:**

<video controls>
  <source src="/uploads/cartpole_v0_output.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

```
Required Score: 50
Average Score Achieved: 140.2
```

Note that everytime you run the code, its not necessary that you get a good score. But if you combine this model with some other algorithm you can make the model more robust.

Thank you for reading this article. If you like it then mention the use cases you think Genetic Algorithm can have in the comment section below. This article was contributed by [Satvik Tiwari](https://www.linkedin.com/in/satvik-tiwari-1a2955155/). Stay tuned for  more Machine Learning stuff.... :)
