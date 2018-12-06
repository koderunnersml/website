---
title: Introduction to Object Detection
date: 2018-12-06T13:06:02.068Z
draft: false
categories: Podcast
tags:
  - machinelearning
  - deeplearning
  - objectdetection
  - singleobjectdetection
author: KoderunnersML
authorImage: uploads/koderunners.jpg
comments: true
share: true
type: post
---
## Image Classification

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

<script src="https://gist.github.com/soumik12345/1b6e65e817a1e3f89addc1495dd83875.js"></script>
