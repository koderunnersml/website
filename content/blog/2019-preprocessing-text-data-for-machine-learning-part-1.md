---
title: 'Preprocessing Text Data for Machine Learning: Part 1'
date: 2019-02-08T14:48:13.194Z
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
### Introduction

In the previous article we discussed various methods to perform [Semantic Slot Filling](http://google.com), a very common problem in the field of Natural Language Processing. We discussed various mathods for tackling such problems such as  Rule Based Approaches and Machine Learning Approaches(including Deep Learning) and also discussed pros and cons of each method. Since Natural Language is a highly unstructured form of data, it needs to be preprocessed a lot to remove dialect-based or idiomatic inconsistancies to attain a state of uniformity and then converted to a mathematical form that can be then used to feed to a Machine Learning Models. In this article we would discuss various methods to preprocess text data.

### Machine Learning with Text Data

Let us consider a Natural Language Problem that can be solved using Machine Learning: **Sentiment Classification**. Sentiment Classification based on user reviews is a very popular application as many businesses all over the world rely on the insights gathered from user reviews to take major decisions. In many cases, such decision-making is automated using **Recommendation Systems**. This means that we would have to create a system that would take as input raw text from user reviews (on a product or service) and output the class of sentiment, usually positive and negative. The possible outputs can be more than two is number or even a range of number if we would treat the problem as a **Regression** problem, but for the sake of simplicity we would consider the problem to be a **Binary Classification** problem. For example,

- *"The watch was very stylish and comfortable to wear and also keeps time accurately"* - is considered to be a positive review.

- *"The band of the watch is loose and it kept loosing time 2 days after getting delivered. Best quality my ass -_-"* - is considered a negative review.

Note that we won't be considering sarcasm a seperate class in this example. So, reviews like *"The headphone is so good that I can listen to music from other galaxies"* or *"I robbed a bank to buy this headphone and now listenning music inside jail"* would be classified positive or negative depending on our model.

### Text Preprocessing

First of all we have to understand what text really is. We can consider text to be a sequence of low-level features like *characters* and *words* or high-level representations such as phrases, named entities or even bigger chunks like sentences, paragraphs, etc. Considering text as a sequence of words might seem a reasonable choice. In fact, we can easily find boundaries of words by splitting sentences by spaces of punctuation.

Defining boundaries of words could be much more difficult in many languages like German where there are ridiculously long compund words that are written without spaces or punctuation. For example, the 63 characters long word *rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz* means *the law for the delegation of monitoring beef labeling*. For the analysis of these words, it might be beneficial to break them up into seperate words

Also, in some languages such as Japanese, there are no spaces.

![Butyoucanstillreaditright???](/uploads/butyoucanstillreaditright.png)

### Tokenization

**Tokenization** is a process that splits an input sequence into several small chunks or **tokens**. You can think of a token as a useful unit for semantic processing. The most important thing to be noted is that a token in case of text need not always be a word, it can be singular characters, words, sentences, paragraphs etc.

Let us see the examples of some popular tokenization methods:

- **Whitespace Tokenizer:** It defines whitespaces as the boundary between tokens. For example,

    ```python
    sentence = "This is an example. I am writing gibberish.\tBy the way, who gave Zack Snyder the idea of Martha? If that's Ben Affleck he sure is crazy."

    white_space_tokens = sentence.split()
    print(white_space_tokens)
    ```

    **Output:**
    ```
    ['This', 'is', 'an', 'example.', 'I', 'am', 'writing', 'gibberish.', 'By', 'the', 'way,', 'who', 'gave', 'Zack', 'Snyder', 'the', 'idea', 'of', 'Martha?', 'If', "that's", 'Ben', 'Affleck', 'he', 'sure', 'is', 'crazy.']
    ```

    The problem in this case is that the tokens "*example*" and "*example.*" are treated differently although they have the same meaning.

- **Word Punct Tokenizer:** It seperates tokens by splitting the input sequence on the basis of punction and whitespace. We can easily implement it using Regular Expressions.

    ```python
    import re

    sentence = "This is an example. I am writing gibberish.\tBy the way, who gave Zack Snyder the idea of Martha? If that's Ben Affleck he sure is crazy."

    tokens = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', sentence)]
    print(tokens)
    ```

    **Output:**
    ```
    ['This', 'is', 'an', 'example', '.', 'I', 'am', 'writing', 'gibberish', '.', 'By', 'the', 'way', ',', 'who', 'gave', 'Zack', 'Snyder', 'the', 'idea', 'of', 'Martha', '?', 'If', 'that', "'", 's', 'Ben', 'Affleck', 'he', 'sure', 'is', 'crazy', '.']
    ```

    The problem in this case is that it splits the word "*that's*" into three seperate tokens `"*that*"`, `"'"`, and `"s"` which is undesirable.

- **Tree Bank Word Tokenizer:** It uses a set of rules or heuristics that defines the grammar of the English language to produce tokenization that actually makes sense for further analysis. It can be implemented using the **NLTK** or **Natural Language Toolkit** in Python.

    ```python
    import nltk

    sentence = "This is an example. I am writing gibberish.\tBy the way, who gave Zack Snyder the idea of Martha? If that's Ben Affleck he sure is crazy."

    tree_bank_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    
    tree_bank_tokens = tree_bank_tokenizer.tokenize(sentence)
    
    print(tree_bank_tokens)
    ```

    **Output:**
    ```
    ['This', 'is', 'an', 'example.', 'I', 'am', 'writing', 'gibberish.', 'By', 'the', 'way', ',', 'who', 'gave', 'Zack', 'Snyder', 'the', 'idea', 'of', 'Martha', '?', 'If', 'that', "'s", 'Ben', 'Affleck', 'he', 'sure', 'is', 'crazy', '.']
    ```

    Tree Bank Tokenizer produces the most meaningful tokens from text in the English Language.

### Token Normalization

**Token Normalization** refers to converting every token into a standard canonical form which it might not have had before. Normalizing Tokens ensures consistency in the data that is further preprocessed or analyzed. There are two processes of Normalizing Tokens:

- **Stemming:** It is the process of removing and replacing suffixes from the token to get the root form of the word which is called a **stem**. Stemming usually refers to heuristics that chop off suffixes.

    **Porter's Stemmer** is the oldest stemmer for the English language. It has five heuristic phases of word reduction applied sequentially. These rules are simple rules like **Regular Grammar** or **Context Free Grammar**. Following are some example of rules in the first phase of Porter's Stemmer:

    | Rule          | Example              |
    | ------------- | -------------------- |
    | `SSES -> SS`  | `caresses -> caress` |
    | `IES -> I`    | `ponies -> poni`     |
    | `SS -> SS`    | `caress -> caress`   |
    | `S -> null`   | `cats -> cat`        |

    ```python
    import nltk

    sentence = "The wolves will not eat the fishes, they will have only lambs."

    tree_bank_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tree_bank_tokens = tree_bank_tokenizer.tokenize(sentence)

    stemmer = nltk.stem.PorterStemmer()
    normalized_tokens_stemming = [stemmer.stem(token) for token in tree_bank_tokens]

    print(normalized_tokens_stemming)
    ```

    **Output:**
    ```
    ['the', 'wolv', 'will', 'not', 'eat', 'the', 'fish', ',', 'they', 'will', 'have', 'onli', 'lamb', '.']
    ```

    The problem with porterstemmer is that due to strict heuristics it often produces irregularities, an example of which is `wolves -> wolv` or `feet -> feet`.

- **Lemmatization:** It is a Token Normalization process that uses liguistic processes to find the base or dictionary form of the word for every token which is also called a *lemma*. Lemmatization utilizes a vocabulary and morphological analysis for finding the lemma of every token.

    The **Wordnet Lemmatizer** is a commonly used lemmatizer implemented in the NLTK library. It uses the [Wordnet Database](https://wordnet.princeton.edu/) to look up lemmas which is a large lexical database of the English language created and maintained by Princeton University.

    ```python
    import nltk

    sentence = "The wolves will not eat the fishes, they will have only lambs."

    tree_bank_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tree_bank_tokens = tree_bank_tokenizer.tokenize(sentence)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    normalized_tokens_lemmatizing = [lemmatizer.lemmatize(token) for token in tree_bank_tokens]

    print(normalized_tokens_lemmatizing)
    ```

    **Output:**
    ```
    ['The', 'wolf', 'will', 'not', 'eat', 'the', 'fish', ',', 'they', 'will', 'have', 'only', 'lamb', '.']
    ```

    Note that Wordnet Lemmatizer tackles some irregularities raised by Porter Stemmer such as `wolves -> wolf` or `feet -> foot`.
    
    Although Wordnet Lemmatizer does a pretty good job, its not 100% accurate. It does a good job normalizing nouns, but might fail for verbs sometimes. Neither Stemming nor Lemmatization is perfect and we need to chose our normalization methodology depending on our track.

- **Normalizing Upper Case Letters:**

    - `Us, us -> us` if both are pronouns but *us*, *US* could also be a pronoun or the name of a country.
    - We could define heuristics to solve this problem:

        - lowercasing beginnning of the sentence
        - lowercasing words in titles
        - leave mid-sentence words as they are

    - We could also use Machine Learning to retrieve true case of tokens which would be quite complex.


In the next article, we will be implementing a simple stemmer from scratch.

This article was contributed by [Soumik Rakshit](https://geekyrakshit.ml/) .

Stay tuned for subsequent articles :)
