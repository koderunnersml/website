---
title: 'Genetic Algorithm: Part 3 - Knapsack Problem'
date: 2019-01-13T11:51:36.636Z
draft: false
categories: Podcast
tags:
  - Machine Learning
  - Knapsack Problem
  - Optimization Techniques
  - Genetic Algorithms
  - geneticalgorithm
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/thinkstockphotos-687644470.jpg
comments: true
share: true
type: post
---
Previously, we discussed about [Genetic Algorithm(GA)](https://koderunners.ml/blog/2018-introduction-to-genetic-algorithm-part-1-intiution/) and its working and also saw its simple [implementation](https://koderunners.ml/blog/2018-introduction-to-genetic-algorithm-part-2-implementation/). 

This time we will solve a classical problem using GA. The problem we will be solving is Knapsack Problem.

# Problem Statement

![](/uploads/thief-2.jpg)

A thief enters a shop carrying knapsack(bag) which can carry 35 kgs of weight. The shop has 10 items, each with a specific weight and price. Now, the thief's dilemma is to make such a selection of items that it maximizes the value(i.e. total  price) without exceeding the knapsack weight. We have to help the thief to make the selection. 

# Solution

We will be using GA to solve this problem. We will follow the same flowchart as we discussed in my first article.

![](/uploads/fig-2.jpg)

 So, here we go....

We begin with randomly initializing the list of items.

**Code:**

```
import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt
```

```
item_number = np.arange(1,11)
weight = np.random.randint(1, 15, size = 10)
value = np.random.randint(10, 750, size = 10)
knapsack_threshold = 35    #Maximum weight that the bag of thief can hold 
print('The list is as follows:')
print('Item No.   Weight   Value')
for i in range(item_number.shape[0]):
    print('{0}          {1}         {2}\n'.format(item_number[i], weight[i], value[i]))
```

**Output:**

```
The list is as follows:
Item No.   Weight   Value
1          3         266

2          13         442

3          10         671

4          9         526

5          7         388

6          1         245

7          8         210

8          8         145

9          2         126

10          9         322
```

Now we declare the initial population. In this problem the idea of chromosome encoding is to have a chromosome consisting as many genes as there are total number of items such that each gene index corresponds to item index in the list. Each gene has a value 1 or 0 which tells whether the corresponding item is present or not.

**Code:**

```
solutions_per_pop = 8
pop_size = (solutions_per_pop, item_number.shape[0])
print('Population size = {}'.format(pop_size))
initial_population = np.random.randint(2, size = pop_size)
initial_population = initial_population.astype(int)
num_generations = 50
print('Initial popultaion: \n{}'.format(initial_population))
```

**Output:**

```
Population size = (8, 10)
Initial popultaion: 
[[0 1 0 1 1 0 0 1 1 1]
 [1 1 1 1 0 1 1 1 0 0]
 [0 1 0 0 0 0 1 1 0 1]
 [0 0 1 0 1 1 0 0 0 0]
 [0 0 1 1 0 0 0 0 0 1]
 [0 1 0 1 1 0 1 0 0 0]
 [1 1 1 0 0 0 1 0 1 0]
 [0 0 0 0 1 1 1 0 0 0]]
```

The fitness function that we will be using for this problem is as follows:

![](/uploads/codecogseqn-5-.gif)

![](/uploads/codecogseqn-7-.gif)

where,   

```
     n = chromosome length
     c_i = ith gene
     v_i = ith value
     w_i = ith weigth 
     kw = knapsack weight
         
```

**Code:**

```
def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int)        
```

Now we select the fittest individuals so that they can undergo crossover.

**Code:**

```
def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents
```

For crossover we will be using one-point crossover(refer to my previous articles). We will be setting crossover rate to a high value to ensure more number  of fittest individuals undergo crossover.

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
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings    
```

In Mutation, which chromosome will undergo mutation is being done randomly. For creating mutants we will be using bit-flip technique i.e. if the selected gene which is going to undergo mutation is 1 then change it to 0 and vice-versa.

**Code:**

```
def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants   
```

As all the necessary functions have been defined so now we will call them in the order of the flow chart to find the required parameters and make all the necessary  initializations.

**Code:**

```
def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, value, population, threshold)      
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history
```

The corresponding items of the parameters in the item_number array will be the ones that the thief will take.

**Code:**

```
parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
selected_items = item_number * parameters
print('\nSelected items that will maximize the knapsack without breaking it:')
for i in range(selected_items.shape[1]):
  if selected_items[0][i] != 0:
     print('{}\n'.format(selected_items[0][i]))
```

**Output:**

```
Last generation: 
[[1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 1 0 0 1 1]
 [1 0 1 1 0 0 0 0 1 1]]

Fitness of the last generation: 
[2156 2156 2156 2156 2156 2156 2156 1911]

The optimized parameters for the given inputs are: 
[array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])]

Selected items that will maximize the knapsack without breaking it:
1

3

4

6

9

10
```

Now we will visualize how the fitness changes with every generation.

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
print(np.asarray(fitness_history).shape)
```

**Output:**

![](/uploads/knap-sack.png)

Thank you for reading this article. This article is contributed by [Satvik Tiwari](https://www.linkedin.com/in/satvik-tiwari-1a2955155/). Stay tuned for more Machine Learning stuff..... :)
