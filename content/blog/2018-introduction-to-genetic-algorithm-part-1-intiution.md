---
title: 'Introduction to Genetic Algorithm: Part 1(Intiution)'
date: 2018-12-20T14:05:49.400Z
draft: false
categories: Podcast
tags:
  - MachineLearning
  - OptimizationTechniques
  - GeneticAlgorithm
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/pic1.jpeg
comments: true
share: true
type: post
---
# Why do we need Genetic Algorithm?

Suppose, we are solving a regression problem in which we have to fit a line across a set of data points having a convex error function. For such problems techniques like Normal Equation and Gradient Descent can easily be used. But what if our function is non-convex?

![](/uploads/fig-1.jpeg)

In the above figure,  if we use the Gradient Descent then we might only be limited to a certain search space as we will be stuck to a local optima. We can randomly go to a particular search space and exploit the information available to reach peak but it may or may not be global maxima. To reach the global maxima we need to explore all the search spaces.

In such cases, _Evolutionary Algorithms (EAs)_ come in handy.

# Intro to Genetic Algorithm

Genetic Algorithm (GA) given by John Holland in 1970  is one of the most popular EAs. It is based on Darwin's theory of "Survival of the Fittest". It is basically used to optimize our problems. In GA we create random changes  to the current solutions through _Selection_, _Crossover_ and _Mutation_ to create new solutions until we reach the best solution. 

# How it works?

It starts with defining an _initial population_ which contains a certain number of solutions. Each solution is called an individual. Each individual solution is encoded as a chromosome which in turn is represented by a set of genes. There are various ways of chromosome encoding which we will discuss later. The figure below gives an idea of how one generation of population looks like.

![](/uploads/whatsapp-image-2018-12-20-at-20.08.13.jpeg)

After chromosome encoding, _fitness_ for every solution is calculated. Higher the fitness, the better is a solution. Fitness is given with the help of a fitness function which is problem specific.

Now, we select the best individuals from this newly created population as parents for the new generation. The higher the fitness value of  a solution the more chances of it are there to be selected as a parent  rendering bad solutions to be left out. These selected individuals go through _crossover_ to create new offspring. By mating the high quality individuals, we can expect to get offsprings better than their parents. By the way, do you know any other algorithm which uses sex so elegantly??? Comment and let us know.

After mating, all the offsprings that we get will contain the same bad characteristics as their parents. To overcome that problem they go through _mutation_ by applying small random changes to their genes thus, creating a diversity in population. These new individuals become the new population or generation. Their fitness is calculated and they go through selection, crossover and mutation. This process goes on and on until we reach the best solution or we complete certain number of generations.

The diagram below gives the whole workflow of GA. 

![](/uploads/fig-2.jpg)

# Chromosome Encoding

The chromosomes are encoded in mainly 3 ways:

1. Binary Encoding: Each chromosome is represented as a set of binary digits.
2. Permutation Encoding: Every chromosome is a string of numbers, which represents number in a sequence. It is mainly used in ordering problems like travelling salesman problem.
3. Value Encoding: In this the actual  value is encoded as it is.

# A bit more about...

## Selection

We select best individuals from previous population for crossover. It is done in many ways. The most common of them are:

1. Fitness Proportionate Selection : In this, each individual can become a parent with a probability which is directly proportional to its fitness.
2. Roulette Wheel Selection: In this we consider a wheel with a fixed point. All the chromosomes of a certain generation occupy some space on the wheel such that the chromosome with greater fitness will get greater pie on the wheel and in turn having a chance of landing in front of fixed point when wheel is rotated.

## Crossover

We combine the genetic material of the selected parent chromosome to produce offspring. Its not necessary that a selected pair of chromosomes will undergo crossover. We define a probabilistic factor called crossover rate which acts as a threshold. This factor decides whether a pair of chromosomes will undergo crossover or not. There are various ways in which we can perform crossover. For example :

Uniform Crossover - In this two individuals create one offspring. We randomly take an individual from the two for every gene of offspring to contribute for the same.

One point Crossover- In this two individuals create two offsprings. We decide a common point for both individuals and their offsprings about which for the first offspring we take all left genes of first parent and all right genes of second parent and vice-versa for second offspring. The figure below gives the idea of one point crossover.

![](/uploads/fig-4.jpeg)

## Mutation

The problem about which we discussed in the beginning gets solved here. Because of mutation, we don't get stuck to a local optima as it introduces genetic diversity in the population. In mutation, we have a mutation rate similar to crossover rate which governs whether an individual will undergo mutation or not. If an individual is selected for mutation then we randomly/uniformly change one/more(as required) genes. For example if our chromosome is binary encoded then we flip the gene value(if it is 0 then we change it to 1 or vice-versa). 

Thank you for reading this. This article was contributed by [Satvik Tiwari](https://www.linkedin.com/in/satvik-tiwari-1a2955155/). Stay tuned for more Machine Learning stuff.
