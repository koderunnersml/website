---
title: Introduction to Object Detection
date: 2018-12-06T15:55:32.175Z
draft: false
categories: Podcast
tags:
  - machinelearning
  - deeplearning
  - objectdetection
  - objdetectionseries
  - rectangle
  - rectangledetector
  - keras
  - tensorflow
  - python
author: KoderunnersML
authorImage: uploads/koderunners.jpg
image: /uploads/obj_detect_header.png
comments: true
share: true
type: post
---
## Image Classification

In Image Classification problems we classify an image to a specific class. The whole image represents one class. We don't want to know exactly where are the object. Usually only one object is presented.
![](/uploads/cat.3.jpg "1")

## Object Detection

Sometimes we need more information from an image rather than just a predicted class. Given an image we want to learn the class of the image and where are the class location in the image. We need to detect a class and a region of interest(usually a rectangle) of where that object is. This is especially useful if a single object is placed in a very small area of an image or multiple objects of same or different classes are present in the image.

![](/uploads/multi-labelled-cats.jpeg)

## Semantic Segmentation

Semantic Segmentation allows us to gather even more information compared to Object Detection. While in Object Detection we usually identify a rectangular region of interest containing a classified object, in Semantic Segmentation we label each pixel in the image with a category label. Semantic Segmentation doesn't differentiate instances, it only cares about individual pixels.
![](/uploads/SemanticSegmentation.png)

## Rectangle Detector

We will be building a very simple object detection module for detecting single rectangles in images. We will be training a simple classifier using **Keras** that will be predicting the bounding boxes of the rectangles. The code used in this article can be found here on this [Kaggle Notebook](https://www.kaggle.com/soumikrakshit/object-detection-single-rectangle).

### Importing The Necessary Libraries

We will be using **Numpy** for linear algebra, **Matplotlib** for visualization, **Scikit-Learn** for splitting the data into training and test set, **Keras** for building the classifier and **Tensorflow** as the backend of Keras.

#### Code:

```
import warnings
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
```

#### Output:

`Using TensorFlow backend.`

### Generating the Dataset

We will be generating 50000 images of height and width 16, where each image will contain an object. Minimum size of an object will be 1 and maximum size will be 8.

We will generate 50000 numpy arrays of shape (16, 16), initialize them with 0 and thus we get the background. Then we will take a random patch from the image and set it to 1. This patch will become our object and the coordinates of the object serve as the bounding boxes.

#### Code:

```
n_images = 50000
n_objects = 1
img_size = 16
min_obj_size = 1
max_obj_size = 8
```

```
def generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size):
    images = np.zeros((n_images, img_size, img_size))
    bounding_boxes = np.zeros((n_images, n_objects, 4))
    for i in range(n_images):
        for j in range(n_objects):
            width, height = np.random.randint(min_obj_size, max_obj_size, size = 2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            images[i, x : x + width, y : y + height] = 1.0
            bounding_boxes[i, j] = [x, y, width, height]
    return (images, bounding_boxes)
```

```
images, bounding_boxes = generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size)
print("Images shape:", images.shape)
print("Bounding Boxes shape:", bounding_boxes.shape)
```

#### Output:

```
Images shape: (50000, 16, 16)
Bounding Boxes shape: (50000, 1, 4)
```

### Visualizing Samples from Generated Images

We will display the image in binary format(0 = White and 1 = Black) and plot a rectangular patch on the image denoting the bounding box.

#### Code:

```
def display_image(index):
    plt.imshow(images[index].T, cmap = "binary", origin='lower', extent = [0, img_size, 0, img_size])
    for box in bounding_boxes[index]:
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2], box[3], ec = 'r', fc = 'none'))
    plt.xticks([])
    plt.yticks([])
    plt.show()
```

```
display_image(np.random.randint(0, n_images))
```

#### Output:

![](/uploads/__results___9_0.png)

### Preprocessing

In the preprocessing step, we will first flatten the images and the bounding boxes, then normalize the images and then split the dataset into training and test set with 33% of the data being in the test set.

#### Code:

```
x = (images.reshape(n_images, -1) - np.mean(images)) / np.std(images)
y = bounding_boxes.reshape(n_images, -1) / img_size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
x.shape, y.shape
```

#### Output:

```
(50000, 256), (50000, 4)
```

### Model Training

We will build a very simple neural network with fully connected layers.

1. The Input Layer consists of 256 nodes, since each image has 256 pixels.
2. The hidden layer consists of 256 nodes, with an activation function ReLu(Rectified Linear Unit).
3. The Output Layer consists of 4 nodes corresponding to the 4 coordinates of a bounding box.

#### Code:

```
def classifier():
    model = Sequential()
    model.add(Dense(256, input_dim = 256))
    model.add(Activation('relu'))
    model.add(Dense(4))
    return model
```

```
model = classifier()
model.summary()
```

#### Output:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
activation_1 (Activation)    (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 1028      
=================================================================
Total params: 66,820
Trainable params: 66,820
Non-trainable params: 0
_________________________________________________________________
```

We will train the network with `adadelta` optimizer and the loss function will be `mean_squared_error`.

#### Code:

```
model.compile(optimizer = "adadelta", loss = 'mean_squared_error', metrics = ['accuracy'])
```

We will implement a callback to store the learning rate at each epoch.

```
learning_rate_history = []
class Learning_Rate_History(Callback):
    def on_epoch_begin(self, epoch, logs = {}):
        learning_rate_history.append(K.get_value(model.optimizer.lr))
        print('Learning Rate:', learning_rate_history[-1])
```

Now we will train the model till 30 epochs.

```
model.fit(x_train, y_train, epochs = 30, validation_split = 0.1, callbacks = [Learning_Rate_History()])
```

#### Output:

```
Train on 30150 samples, validate on 3350 samples
Epoch 1/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 63us/step - loss: 0.0386 - acc: 0.6658 - val_loss: 0.0257 - val_acc: 0.7651
Epoch 2/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 0.0058 - acc: 0.8290 - val_loss: 0.0042 - val_acc: 0.8490
Epoch 3/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 0.0026 - acc: 0.8766 - val_loss: 0.0031 - val_acc: 0.8215
Epoch 4/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 0.0019 - acc: 0.8901 - val_loss: 0.0019 - val_acc: 0.8991
Epoch 5/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 0.0015 - acc: 0.9048 - val_loss: 0.0016 - val_acc: 0.8669
Epoch 6/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 53us/step - loss: 0.0013 - acc: 0.9059 - val_loss: 0.0012 - val_acc: 0.8940
Epoch 7/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 0.0011 - acc: 0.9066 - val_loss: 0.0014 - val_acc: 0.8764
Epoch 8/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 9.7049e-04 - acc: 0.9162 - val_loss: 0.0015 - val_acc: 0.8919
Epoch 9/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 8.9787e-04 - acc: 0.9156 - val_loss: 0.0011 - val_acc: 0.9107
Epoch 10/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 8.0017e-04 - acc: 0.9162 - val_loss: 9.8927e-04 - val_acc: 0.9364
Epoch 11/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 7.3458e-04 - acc: 0.9198 - val_loss: 6.2795e-04 - val_acc: 0.9325
Epoch 12/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 6.7894e-04 - acc: 0.9202 - val_loss: 6.7405e-04 - val_acc: 0.9063
Epoch 13/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 6.3128e-04 - acc: 0.9209 - val_loss: 8.1824e-04 - val_acc: 0.9304
Epoch 14/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 5.7864e-04 - acc: 0.9208 - val_loss: 5.4598e-04 - val_acc: 0.9534
Epoch 15/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 5.3993e-04 - acc: 0.9206 - val_loss: 6.0319e-04 - val_acc: 0.9099
Epoch 16/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 5.1695e-04 - acc: 0.9234 - val_loss: 4.6435e-04 - val_acc: 0.9242
Epoch 17/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 4.7829e-04 - acc: 0.9239 - val_loss: 7.4663e-04 - val_acc: 0.9239
Epoch 18/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 4.5803e-04 - acc: 0.9237 - val_loss: 6.2423e-04 - val_acc: 0.9110
Epoch 19/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 4.3221e-04 - acc: 0.9238 - val_loss: 4.8555e-04 - val_acc: 0.9316
Epoch 20/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 53us/step - loss: 4.1499e-04 - acc: 0.9235 - val_loss: 4.5414e-04 - val_acc: 0.9469
Epoch 21/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.8647e-04 - acc: 0.9258 - val_loss: 5.5298e-04 - val_acc: 0.9116
Epoch 22/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.7609e-04 - acc: 0.9251 - val_loss: 4.3545e-04 - val_acc: 0.9269
Epoch 23/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.4303e-04 - acc: 0.9255 - val_loss: 5.7299e-04 - val_acc: 0.9218
Epoch 24/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.3132e-04 - acc: 0.9272 - val_loss: 8.2355e-04 - val_acc: 0.9173
Epoch 25/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.1925e-04 - acc: 0.9266 - val_loss: 5.7306e-04 - val_acc: 0.9230
Epoch 26/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 3.0340e-04 - acc: 0.9280 - val_loss: 4.8713e-04 - val_acc: 0.9364
Epoch 27/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 51us/step - loss: 2.9643e-04 - acc: 0.9249 - val_loss: 3.5696e-04 - val_acc: 0.8884
Epoch 28/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 2.9189e-04 - acc: 0.9269 - val_loss: 3.7836e-04 - val_acc: 0.9555
Epoch 29/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 2.7267e-04 - acc: 0.9261 - val_loss: 3.0140e-04 - val_acc: 0.9099
Epoch 30/30
Learning Rate: 1.0
30150/30150 [==============================] - 2s 52us/step - loss: 2.6856e-04 - acc: 0.9250 - val_loss: 3.7276e-04 - val_acc: 0.9469
<keras.callbacks.History at 0x7f5d9e326a20>
```

### Prediction and Visualization on Test Set

Now we will be get the predicted bounding boxes on the Test set.

#### Code:

```
y_pred = model.predict(x_test)
box_pred = y_pred * img_size
box_pred.shape
```

#### Output:

```
(16500, 4)
```

Let us visualize the predicted bounding boxes on test data.

#### Code:

```
def display(x, box):
    index = np.random.randint(0, len(x))
    plt.imshow(x[index].reshape(16, 16).T, cmap = 'binary', origin = 'lower', extent = [0, img_size, 0, img_size])
    plt.gca().add_patch(Rectangle((box[index][0], box[index][1]),
                                      box[index][2], box[index][3],
                                      ec = 'r', fc = 'none'))
    plt.xticks([])
    plt.yticks([])
    plt.show()
```

```
display(x_test, box_pred)
```

#### Output:

![](/uploads/__results___30_0.png)

This article was contributed by [Soumik Rakshit](https://geekyrakshit.ml/).

Thank you for reading and stay tuned for subsequent articles :)
