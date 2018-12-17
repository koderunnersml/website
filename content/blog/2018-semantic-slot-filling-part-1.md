---
title: 'Semantic Slot Filling: Part 1'
date: 2018-12-17T13:32:32.343Z
draft: false
categories: Podcast
tags:
  - machinelearning
  - naturallanguageprocessing
  - nlp
  - contextfreegrammar
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/nlp.png
comments: true
share: true
type: post
---
## Semantic Slot Filling: Part 1

One way of making sense of a piece of text is to tag the words or tokens which carry meaning to the sentences. In the field of Natural Language Processing, this problem is known as **Semantic Slot Filling**. There are three main approaches:

1. Rule Based Approaches
2. Machine Learning Approaches
3. Deep Learning Approaches

Let us consider the following query text:
> **Show me all the Buses from Kolkata to Bhubanshwar on Friday.**

Given this piece of text, we have to find some slots which may be Destination, City, Date etc. We will see how to fill these slots using the previously mentioned approaches.

### 1. Rule Based Approach
This approach consists on **Semantic Slot Filling** techniques using **Regular Grammars** or **Context Free Grammars**.

A **Grammar** is defined as a set of production rules which are used to generate strings of a language, which in this case may be a **Regular** or a **Context Free Language**.

In this case we would use a Context Free Grammar(CFG) to do Semantic Slot Filling. Effectively we can always chose to use Context Free Grammars since they are a super-set of Regular Grammar.

Let us consider the following CFG:
```
S -> SHOW BUSES ORIGIN DESTINATION DATE|...
SHOW -> show me|i want|can i see|...
BUSES -> bus|a bus|buses
ORIGIN -> from CITY
DESTINATION -> to CITY
CITY -> kolkata|bhubaneshwar|Ahmedabad|...
DATE -> sunday|monday|...|saturday
```
Note that the Non-terminal strings in this case are the slots. Let us create the Parse tree of the above grammar

![Parse Tree](/uploads/parse_tree.png)

Now let us derive our query text from the parse tree

![Parse Tree on Query](/uploads/parse_tree_on_query.png)

Since we can use this Context Free Grammar to parse our query string, we would know exactly which non-terminal strings produced the terminal tokens in the query string and we can tag the tokens in our string accordingly. This would result in the following tagging:

```
<SHOW>Show me</SHOW>
all the <BUSES>Buses</BUSES>
<ORIGIN>from <CITY>Kolkata</CITY></ORIGIN>
<DESTINATION>to <CITY>Bhubanshwar</CITY></DESTINATION>
on <DATE>Friday</DATE>.
```

**Advantage:** : This approach has very high precision.

**Disadvantage:**
1. Someone(usually a linguist) would have to write down the rules manually which is very time consuming.
2. The recall of this process will not be good because, it would not be practical to write down every possible date.


### 2. Machine Learning Approach
For a machine learning approach to solve this problem, we need some kind of data to learn from. This data usually comes in the form of a **Training Corpus** which is a large body of text with the necessary tags present in the form of Markup. For example,

```
Are any <BUSES>buses</BUSES> leaving <DESTINATION>for <CITY>Kolkata</CITY></DESTINATION> <DATE>today</DATE>?
```

Once we have our training data, we have to do some feature engineering, and extract some useful features such as

- Is the word in uppercase?
- Is the word present in the name of cities?
- What are the previous words?
- What are the next words?
- What is the previous slots?

and so on....

Feature engineering usually depends on the training corpus, the application at hand (which in this case is tagging words) and most importantly the creativity of the engineer.

Now we need to define our model. The model may be a probabilistic model that gives the probability of the tags associated with a given word. This probability will depend of the features representing some text and some **parameters**.

```
model -> p(tags|words) -> function of features and parameters
```

The parameters of the model should be trained. So we will need to take your train data and fit the model to this data and maximize the probability of what we see, by the parameters. Once we have the trained parameters, we can take the feature corresponding to a tag and infer that the tag with the maximum probability is associated with the word.

```
predicted_tag = argmax(p(tags|words))
```

### 3. Deep Learning Approach
The Deep Learning approach is largely similar to the machine learning approach except in the Deep Learning methodology, features do not need to be manually engineered, rather we feed an encoded sequence of words into a **Neural Network** and the different hidden layers of the Neural Network act as feature extractors.

Usually for Semantic Slot Filling and other Sequence Related Tasks, the most popular Deep Learning Models are **Recrrent Neural Networks**(RNNs).

![RNN Architecture](/uploads/rnn.png)

For practical purposes, **Gated Recurrent Units or GRU** blocks and **Long Short Term Memory or LSTM** blocks are used in the RNN architecture.

This article was contributed by [Soumik Rakshit](https://geekyrakshit.ml/)
Stay tuned for subsequent articles :)
