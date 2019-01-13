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
image: /uploads/pic2.jpg
comments: true
share: true
type: post
---
In Part 1 of Genetic Algorithm, we discussed about Genetic Algorithm and its workflow. Now its time for its implementation.

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
import matplotlib.pyplot as plt
```

Now, we create an initial population using _decimal representation. _So our population is basically a set of random numbers. Number of generations is problem specific and so is the solutions per population. For the ease of understanding we are taking 5 generations and 8 solutions per population.

**Code:**

```
inputs = np.asarray([4,10,-8,0,5,-3])
solutions_per_pop = 8
pop_size = (solutions_per_pop, inputs.shape[0] )
initial_population = np.random.uniform(low = -3.0, high = 3.0, size = pop_size)
num_generations = 5
print('Initial population: \n{}'.format(initial_population))
```

**Output:**

```
Initial population: 
[[-2.93328108  0.15536064 -1.80639808  0.80643693 -2.72261661 -1.22164978]
 [ 0.31280087 -0.09681246 -2.64161295  0.03947202  1.58390633 -2.87012794]
 [ 2.17724501 -2.28752884 -0.97660438 -2.70319968 -2.42832452 -0.91756842]
 [-1.50904547 -0.91224752 -2.0727139  -0.06819324 -0.12204797  2.35770583]
 [ 2.71096926  2.60392142  0.43119847  1.50794491 -1.99507966  0.26095386]
 [-2.04712335 -2.04704483 -2.85673835 -0.98028334 -0.13509991 -1.3988682 ]
 [-2.51611324  2.36575384 -0.12775799 -0.95871885 -1.88232301 -0.20386567]
 [-0.91357096  0.46730088 -1.48259089  0.69565825 -0.18168019 -1.65393245]]
```

Everytime we run the code the population will be different because we are using random numbers to do so.

Now, we calculate the fitness. In this implementation the fitness function that we will be using is :

![](/uploads/fitness_function.gif)

where, 

> w = weights
>
> x = inputs
>
> n = number of solutions per population                                                                                                                                                                                                                                                   

**Code:**

```
def cal_fitness(inputs, population):
    fitness = np.sum(population * inputs, axis=1)
    return fitness
```

In selection, for every iteration we select the fittest individual and add it to the parents.

**Code:**

```
def selection(population, fitness, num_parents):    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_index = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_index[0][0],:]
        fitness[max_fitness_index[0][0]] = -9999999
    return parents
```

Now, in crossover we perform _one point crossover._ We declare a _crossover point_ such that it is a middle point and a _crossover rate_ which is set to a high value. For every two individuals a random number between 0 and 1 is generated. If the number is less than or equal to crossover rate then, the individuals are mated. For every two individual one offspring is created.

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
    weights, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(inputs, population)
        fitness_history.append(fitness)
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
    return weights, fitness_history
```

The fittest individual is the required solution i.e., the _weights_ of our equation.

**Code:**

```
weights, fitness_history = optimize(inputs, initial_population, pop_size, num_generations)
print('The optimized weights for the given inputs are: \n{}'.format(weights))
```

**Output:**

```
Last generation: 
[[-1.7797316   2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-2.56064005  2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-2.56064005  2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-2.56064005  2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-2.56064005  2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-2.56064005  2.9122713  -1.74434141  0.9858133   0.06528816 -2.98698004]
 [-1.4456499   2.99217208 -0.23599195 -1.27945883  1.70275531 -2.52328275]
 [-1.4456499   2.99217208 -0.23599195 -1.27945883  1.70275531 -2.52328275]]

Fitness of the last generation: 
[45.24589885 42.12226504 42.12226504 42.12226504 42.12226504 42.12226504
 42.1106816  42.1106816 ]

The optimized weights for the given inputs are: 
[array([-1.7797316 ,  2.9122713 , -1.74434141,  0.9858133 ,  0.06528816,
       -2.98698004])]
```

Time to see how fitness varies with every generation.

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

**Visualization:**

![](/uploads/ga-2-new-plot.png)

Thank you for reading this article. This article is contributed by [Satvik Tiwari](https://www.linkedin.com/in/satvik-tiwari-1a2955155/). Stay tuned for more Machine Learning stuff....  :)
