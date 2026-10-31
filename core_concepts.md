# Neuron
A small calculation part of a whole network is called **neuron**. A network is built using these tiny neurons in a large quantity.

Lets take logistic regression as and example here. lets see the way of its calculation in deep learning here, and understand how neuron looks in simple words.

```text
                                      value < 0.5  = person will not buy insurance
                                      value >= 0.5 = person will buy insurance

  Age = 35
      |
      v
      +-----------------------------+-----------------------------+
      |                             |                             |
      |       y = 0.042 * x - 1.53  |         1                   |
      |                             |     z = ----------------    |
      |                             |         1 + e^(-y)          |
      |                             |                             |
      +-----------------------------+-----------------------------+
                                                                  |
                                                                  v
                                                                0.48
```

see this, we pass the age 35, use the linear function then pass the output of the linear function to sigmoid function to get a output between 0 and 1 and where we got 0.45.

**This is a simple network with one input and one output, and the whole block in the middle is called a _neuron_**

Now if we have some more inputs like income and education. we have this below.

![simple visualization of a neuron](images\neuron.png)

now the values in there maybe random but the weights from the input and bias get updated during the __backpropogation__.

---

# Neural network
Now that we got to know about what is neuron right, now lets see how a actual neural network looks like in words and there is also a image to see that. **a simple network contains bunch of neurons stacked on top of each other, then we have something named as hidden layer which has a crucial role and then a an output layer, this is a simple network with one hidden layer.** Things which can vary based one situation is number or neurons per layer and number of hidden layers in the network. There is also some variation related to the connections between the neurons which is another concept under types or networks.

![simple visualization of a network](images\network.png)


There is some lot of math behind a neural network which needs to be understood first side by side while learning. 

---