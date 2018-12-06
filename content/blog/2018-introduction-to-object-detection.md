---
title: Introduction to Object Detection
date: 2018-12-06T15:55:32.175Z
draft: true
categories: Podcast
tags:
  - machinelearning
  - deeplearning
  - objectdetection
author: KoderunnersML
authorImage: uploads/koderunners.jpg
comments: true
share: true
type: post
---
## Image Classification

## 

In Image Classification problems we classify an image to a specific class. The whole image represents one class. We don't want to know exactly where are the object. Usually only one object is presented.

![Cat](/uploads/cat.3.jpg)

## Object Detection

Sometimes we need more information from an image rather than just a predicted class. Given an image we want to learn the class of the image and where are the class location in the image. We need to detect a class and a region of interest(usually a rectangle) of where that object is. This is especially useful if a single object is placed in a very small area of an image or multiple objects of same or different classes are present in the image.

![Multi Labelled Cats](/uploads/multi-labelled-cats.jpeg)

## Semantic Segmentation

Semantic Segmentation allows us to gather even more information compared to Object Detection. While in Object Detection we usually identify a rectangular region of interest containing a classified object, in Semantic Segmentation we label each pixel in the image with a category label. Semantic Segmentation doesn't differentiate instances, it only cares about individual pixels.

![](/uploads/semanticsegmentation.png)

## Rectangle Detector

We will be building an object detection module for detecting single rectangles in images. We will be training a simple classifier using **Keras** that will be predicting the bounding boxes of the rectangles.

### Importing The Necessary Libraries

We will be using **Numpy** for linear algebra, **Matplotlib** for visualization, **Scikit-Learn** for splitting the data into training and test set, **Keras** for building the classifier and **Tensorflow** as the backend of Keras.

#### Code:

<pre><code>import warnings
# Ignoring unnecessary warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
# For visualizing Bounding Boxes
from matplotlib.patches import Rectangle


import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback

# For visualizing the classifier
from IPython.display import SVG

# For splitting the Dataset
from sklearn.model_selection import train_test_split
</code></pre>

#### Output: <pre><code>Using TensorFlow backend.</code><pre>

### Generating The Dataset

We will be generating 50000 images of height and width 16, where each image will contain an object. Minimum size of an object will be 1 and maximum size will be 8.

<pre><code>n_images = 50000
n_objects = 1
img_size = 16
min_obj_size = 1
max_obj_size = 8</code><pre>

<pre><code>def generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size):
    images = np.zeros((n_images, img_size, img_size))
    bounding_boxes = np.zeros((n_images, n_objects, 4))
    for i in range(n_images):
        for j in range(n_objects):
            width, height = np.random.randint(min_obj_size, max_obj_size, size = 2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            images\\\[i, x : x + width, y : y + height] = 1.0
            bounding_boxes\\\[i, j] = \\\[x, y, width, height]
    return (images, bounding_boxes)</code><pre>

<pre><code>images, bounding_boxes = generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size)
print("Images shape:", images.shape)
print("Bounding Boxes shape:", bounding_boxes.shape)</code><pre>

<h4>Output: <pre><code>Images shape: (50000, 16, 16)
Bounding Boxes shape: (50000, 1, 4)</code><pre></h4>

<h3>Visualizing Generated Samples</h3>
<pre><code>def display_image(index):
    plt.imshow(images\\[index].T, cmap = "binary", origin='lower', extent = \\[0, img_size, 0, img_size])
    for box in bounding_boxes\\[index]:
        plt.gca().add_patch(Rectangle((box\\[0], box\\[1]), box\\[2], box\\[3], ec = 'r', fc = 'none'))
    plt.xticks(\\[])
    plt.yticks(\\[])
    plt.show()
</code><pre>
<pre><code>display_image(np.random.randint(0, n_images))</code><pre>

<h4>Output:</h4>
<img src="/uploads/__results___9_0.png">
