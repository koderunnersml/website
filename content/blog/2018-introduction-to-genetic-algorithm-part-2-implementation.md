---
title: 'Genetic Algorithm: Part 2 - Implementation'
date: 2018-12-20T15:27:08.533Z
draft: false
categories: Podcast
tags:
  - MachineLearning
  - GeneticAlgorithmImplementation
  - OptimizationTechniques
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/pic-1.jpg
comments: true
share: true
type: post
---
In my previous article, we discussed about Genetic Algorithm and its workflow. Now its time for its implementation.

Lets consider an equation:

> **Y = w1x1 + w2x2 + w3x3 + w4x4 +w5x5 +w6x6**

Given the inputs (x1, x2, x3, x4, x5, x6)=(4, 10, -8, 0, 5, -3) we have to find the weights w1, w2, w3, w4, w5, w6 such that it maximizes the output equation. So, we will use GA to find the weights.

We are going to follow the same flowchart of GA as was in previous article.

![](/uploads/fig-2.jpg)

We first import the necessary libraries.

```
import numpy as np
import random as rd
from random import randint  
```

Now, we create an initial population using _decimal representation. _So our population is basically a set of random numbers. Number of generations is problem specific and so is the solutions per population. For the ease of understanding we are taking 5 generations and 8 solutions per population.

**Code:**

```
inputs = np.asarray([4,10,-8,0,5,-3])
solutions_per_pop = 8
pop_size = (solutions_per_pop, inputs.shape[0] )
initial_population = np.random.uniform(low = -3.0, high = 3.0, size = pop_size)
num_generations = print('Initial population: \n{}'.format(initial_population))
```

**Output:**

```
Initial population: 
[[-1.01245739 -1.56646273  0.98240416  1.56719941 -0.88343664 -2.97975112]
 [-0.26446    -0.63331283  2.01506987  1.55716117 -0.73383902  2.66251002]
 [ 0.51937074 -1.60662555  2.18967848 -0.9790294  -0.33791569 -2.80875267]
 [-0.81239689 -1.18476129  1.12083248  2.58912426  0.70442334 -2.86601968]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-0.1596207  -1.98807885 -2.06621765  1.7338106  -1.99176258  0.02053455]
 [-1.86821137 -1.938959    2.26166578  1.7462855   0.78947019 -1.58978935]
 [-2.54771976 -2.05911834 -0.92403029 -0.31943213 -1.94437704  2.68513142]]
```

Everytime we run the code the population will be different because we are using random numbers to do so.

Now, we calculate the fitness. In this implementation the fitness function that we will be using is :

![](/uploads/fitness_function.gif)

where, 

> w = weights

> x = inputs

> n = number of solutions per population                                                                                                                                                                                                                                                   

**Code:**

```
def cal_fitness(inputs, population):
    fitness = np.sum(population*inputs, axis=1)
    return fitness
```

In selection, for every iteration we select the fittest individual and add it to the parents.

**Code:**

```
def selection(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_index = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_index[0][0],:]
        fitness[max_fitness_index[0][0]] = -9999999
    return parents
```

Now, in crossover we perform _one point crossover._ We declare a _crossover point_ such that it is a middle point and a _crossover rate_ which is set to high a high point. For every two individuals a random number between 0 and 1 is generated. If the number is less than or equal to crossover rate then, the individuals are mated. For every two individual one offspring is created.

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

For mutation, we set mutation rate to a small value so that few individuals got though mutation. For every offspring we generate a random number between 0 and 1. If its less than or equal to the mutation rate then that individual goes through mutation.

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
        mutants[i,int_random_value] += np.random.uniform(-1.0, 1.0, 1)  
    return mutants
```

In the function below we first decide how many parents and offsprings will be there. And then call all the functions in the order of the flow chart that we discussed earlier. In the end we calculate the fitness of the last generation and then find the fittest individual among them.

**Code:**

```
def optimize(inputs, population, pop_size, num_generations):
    weights = []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(inputs, population)
        parents = selection(population, fitness, num_parents)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(inputs, population)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    weights.append(population[max_fitness[0][0],:])
    return weights    
```

The fittest individual is the required solution i.e., the _weights_ of our equation.

**Code:**

```
weights = optimize(inputs, initial_population, pop_size, num_generations)
print('The optimized weights for the given inputs are: \n{}'.format(weights))
```

**Output:**

```
Last generation: 
[[-0.97975558 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -3.21160968 -0.83843799  1.68509728 -1.52362102]
 [-1.57231087 -0.43209536 -2.86624978 -0.83843799  1.68509728 -1.52362102]
 [-0.81239689 -1.18476129  0.22196107  2.58912426  0.70442334 -2.86601968]]

Fitness of the last generation: 
[ 115.32727417  112.95705303  112.95705303  112.95705303  112.95705303
  119.17353134  112.95705303   17.68249277]

The optimized weights for the given inputs are: 
[array([-1.57231087, -0.43209536, -3.21160968, -0.83843799,  1.68509728,
       -1.52362102])]
```





Thank you for reading this article. This article is contributed by [Satvik Tiwari](https://www.linkedin.com/in/satvik-tiwari-1a2955155/). Stay tuned for more Machine Learning stuff....  :)
