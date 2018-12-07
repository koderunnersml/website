---
title: Intersection Over Union
date: 2018-12-07T13:28:12.474Z
draft: false
categories: Podcast
tags:
  - machinelearning
  - deeplearning
  - objectdetection
  - python
  - tensorflow
  - keras
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/cat2_2.jpg
comments: true
share: true
type: post
---
In the previous article, Indroduction to Object Detection, we have seen how a single object can be detected in an image by predicting a bounding box for the object. Like all Machine Learning Tasks, prediction of bounding box requires an accuracy metric to tell us how accurate the predictions are. In this article we will be discussing an accuracy metric that can be used for Object Detection.

## I Owe You
Let us consider the following photo of a grumpy cat.

![](/uploads/cat2.jpg)

Now, in ideal case, an object detection algorithm should should be identifying him somewhat like this.

![](/uploads/cat2_1.jpg)

But, if an object detection algorithm outputs the following blue bounding box, how do we tell how much off the mark our prediction is???

![](/uploads/cat2_2.jpg)

In this case, we calculate **IOU** or **Intersection** Over Union for the following bounding boxes.

![](/uploads/IOU.jpg)

**`IOU(Box1, Box2) = Intersection_Size(Box1, Box2) / Union_Size(Box1, Box2)`**

## Implementing IOU

The bounding box coordinates are in the form `(x, y, width, height)`. We will first calculate the width and height of the Intersection Box and size of Intersection will be area of the Intersection Box. We can get the Union size by subtracting the Intersection size from total area. All the code is part of the following [Kaggle Notebook](https://www.kaggle.com/soumikrakshit/object-detection-single-rectangle).

#### Code:
```
def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U
```
```
iou = [IOU(y_test[i], y_pred[i]) for i in range(len(x_test))]
```

We will modify the visualization code to also show the respective IOU.

```
def display(x, box, box_pred):
    index = np.random.randint(0, len(x))
    plt.imshow(x[index].reshape(16, 16).T, cmap = 'binary', origin = 'lower', extent = [0, img_size, 0, img_size])
    plt.gca().add_patch(Rectangle((box_pred[index][0], box_pred[index][1]),
                                      box_pred[index][2], box_pred[index][3],
                                      ec = 'r', fc = 'none'))
    plt.title("IOU: " + str(iou[index]))
    plt.xticks([])
    plt.yticks([])
    plt.show()

```
```
display(x_test, y_test, box_pred)
```

#### Output:
![](/uploads/__results___30_1.png)

This article was contributed by [Soumik Rakshit](https://geekyrakshit.ml)

Thank you for reading and stay tuned for subsequent articles :)
