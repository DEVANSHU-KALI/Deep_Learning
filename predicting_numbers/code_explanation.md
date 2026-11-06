This file covers the whole process of how is the code working and other thing related to the code.

# How are the images passed as input to the network. 
- each image has a dimension of 28x28, which you can see the notebook. each pixel holds the color ranging from 0-255. now that image is converted to that grid containing all those values. now comes the interesting part, **that 2d matrix is converted to 1D, simply knowns as flatening the matrix, which becomes a single line of values, and if its 28x28, we will have 784 neurons where each holds a single value of that grid.** here is a image of that. 

![showing how image is passed as input](..\images\network_2.png)

## what type of data are we using for this network
- we are going to use mninst hand drawn digits dataset, which has 70000 images, covering all type of digits we have. 
---

# initializing of model and how to pass input.
The fact about neural networks is, **there is nothing like input layer**, most of the images of networks might feel like there is input layer and it does something. **That thing naming input layer is not actually a functioning layer there**. when its about passing data to the network, we only specify that this will be your input. 

A layer is consider is a actual functioning layer only when it does the following both:
- Parameters: Learnable weights ($W$) and biases ($b$).
- Computation: Matrix multiplications, additions, and non-linear mappings. 

so you need to remember that, when we say input layer, its just a name there, not a functional layer. **Hidden layer and output layer are the one which are actual layers**, because they do some actual work as mentioned below.

## now coming to the code
```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),               # 1. Doorway: receives 784 inputs
    layers.Dense(10, activation='sigmoid')    # 3. Output layer: predicts 10 classes
])
```
this is basic MLP with 1 input and 1 output layer. we use keras to initialize the model. 

`kersa.Sequential([])`: is most used simple way to build the model, and it suits our solution. there are other two types we can used named as "Functional API" amd "Model Subclassing", which are used based on specific conditions. I'll cover about the other ways if needed.

`layers.Input(shape=())`: this tells the model that, this is your input. 784 values.  If you want to go deep and understand more about this syntax and what other things it takes, you can go and Visit ["Tensorflow"](https://www.tensorflow.org/api_docs/python/tf/keras/Input). ctrl+left_click on this to go to the page.

`layers.Dense()`: output layer. we here passed the number of neurons it should have and the activation function. activation functions will be explained later. If you want to understand more about this specific syntax and other things, Visit ["Tensorflow"](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Den).

This is a simple way of initializing the model.

## after initialization.
- the very next step is to compile the model, i.e, initialize the hyperparameters. you can see the code. if you want to know more about .compile(), visit ["Keras"](https://keras.io/api/models/model_training_apis/).
- next we have fit method, where you actually pass the data to the network, similarly done in ml. to know more visit ["Keras"](https://keras.io/api/models/model_training_apis/), scroll the page down to see info related to this fit method.

# evaluation 
- even without evaluate method, you can see the accuracy and loss directly when you run that training cell, but we do have some other things which we can pass as parameters. you can visit ["Keras"](https://keras.io/api/models/model_training_apis/) and scroll down to ses info about this method. 

---
# info related to flattening the data.
You only flatten images at the very beginning when feeding them into __Dense (Fully Connected / MLP)__ layers.

Modern computer vision architectures, like __CNNs (Convolutional Neural Networks)__ and __Vision Transformers__, do not flatten the 2D image at the start—doing so would actually destroy the image's structure.

## 1. Why MLPs Require Flattening
Dense layers (keras.layers.Dense) only understand 1D vectors:
$$\text{Output} = W \cdot x + b$$

A matrix multiplication expects $x$ to be a single column of numbers. It has no concept of rows, columns, height, or width. Because a Dense layer physically cannot accept a 2D matrix like (28, 28), you are forced to squash it into a 1D vector of 784 values.
## 2. The Big Problem with Flattening Images
When you flatten an image:
- Pixels that are vertically next to each other (e.g., Row 1 Col 5 and Row 2 Col 5) get separated by hundreds of indices in the 1D array.
- The model loses spatial locality—it forgets that pixel $(x, y)$ is directly above pixel $(x, y+1)$.
- The model must re-learn spatial geometry from scratch using brute-force weights.

# Loss function and optimizer
- You need to first know what is loss and optimizer.   what are the different loss functions used, if you came through ML, you would've known this but in DL there are some more new type of loss functions which are not seen in ML.
- the one which we used is `sparse_categorical_crossentropy`. used when there are two or more label classes. We expect labels to be provided as integers. There are lot of other loss functions which can be used based on the different conditions, to know more about different loss functions visit ["Keras"](https://keras.io/api/losses/) 
- same goes for optimizers, visit ["Keras"](https://keras.io/api/optimizers/) to know about different optimizers and also about the one which we used in the code.

