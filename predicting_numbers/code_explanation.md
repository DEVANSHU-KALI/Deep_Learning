This file covers the whole process of how is the code working and other thing related to the code.

# How are the images passed as input to the network. 
- each image has a dimension of 28x28, which you can see the notebook. each pixel holds the color ranging from 0-255. now that image is converted to that grid containing all those values. now comes the interesting part, **that 2d matrix is converted to 1D, simply knowns as flatening the matrix, which becomes a single line of values, and if its 28x28, we will have 784 neurons where each holds a single value of that grid.** here is a image of that. 

![showing how image is passed as input](images/network_2.png)

## what type of data are we using for this network
- we are going to use mninst hand drawn digits dataset, which has 70000 images, covering all type of digits we have. 
---

need to talk about 
1. info related to "passing input in the code" including "why not .Dense while configuring input"
2. info related to flattening the train set. including why 
3. loss function we used
4. explaining the confusion matrix
