This file covers the whole process of how is the code working and other thing related to the code.

# How are the images passed as input to the network. 
- each image has a dimension of 28x28, which you can see the notebook. each pixel holds the color ranging from 0-255. now that image is converted to that grid containing all those values. now comes the interesting part, **that 2d matrix is converted to 1D, simply knowns as flatening the matrix, which becomes a single line of values, and if its 28x28, we will have 784 neurons where each holds a single value of that grid.** here is a image of that. 

![showing how image is passed as input](images/network_2.png)

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

need to talk about 
2. info related to flattening the train set. including why 
3. loss function we used
4. explaining the confusion matrix
