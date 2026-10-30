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

![alt text](images\neuron.png)

now the values in there maybe random but the weights from the input and bias get updated during the __backpropogation__.

---

