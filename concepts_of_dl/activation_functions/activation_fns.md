# what are activation functions
- An activation function is a mathematical gate that determines whether a specific neuron in a neural network should "fire" and pass its information forward to the next layer. It takes the raw, calculated output of a neuron and transforms it into a standardized format before sending it deeper into the network.

To understand why they are strictly necessary, you have to look at the fundamental math happening inside the node.

## The Linear Problem
When a neuron receives data, it performs a basic linear operation: it multiplies the inputs by their assigned weights and adds a bias ($z = Wx + b$).

If you stack dozens of neural layers that only do this linear math, the entire network mathematically collapses into one giant linear regression model. Because linear models can only draw straight lines through data, a network without activation functions would find it biologically and mathematically impossible to learn complex, real-world patterns like recognizing the curves of a handwritten digit, mapping human speech, or identifying a stop sign in a video feed.

## The Non-Linear Solution
Activation functions solve this by introducing non-linearity. By passing that raw linear number ($z$) through a non-linear mathematical filter, the network gains the ability to bend, curve, warp, and fold its decision boundaries to perfectly wrap around highly complex data.

## different activation functions
**ReLU (Rectified Linear Unit)**

* **What it is:** The standard, default non-linear activation function used in the vast majority of modern neural networks.
* **Its uses:** Used almost exclusively in the hidden layers of networks (like MLPs and CNNs). It is preferred because it is computationally highly efficient and solves the "vanishing gradient" problem for positive values, allowing deep networks to train faster.
* **How it works:** It acts as a strict mathematical threshold. If the incoming calculation is a negative number, it blocks it and outputs $0$ (turning the neuron off). If the number is positive, it passes that exact number forward unchanged.
* **Formula:** $f(x) = \max(0, x)$

**Leaky ReLU**

* **What it is:** A modified version of the standard ReLU function designed specifically to fix the "Dying ReLU" problem.
* **Its uses:** Used in hidden layers as a drop-in replacement when a network using standard ReLU suffers from dead neurons and flatlines. It is also highly preferred in complex architectures like Generative Adversarial Networks (GANs).
* **How it works:** Positive numbers pass through unchanged, exactly like standard ReLU. However, instead of crushing negative numbers to a flat $0$, it multiplies them by a very small constant parameter (typically $0.01$). This creates a slight downward slope, allowing a tiny mathematical gradient to survive so the neuron can continue updating its weights.
* **Formula:** $f(x) = \max(0.01x, x)$

**Sigmoid**

* **What it is:** A mathematical function that maps any incoming numerical value to a decimal strictly between $0$ and $1$.
* **Its uses:** Primarily used in the final output layer of a neural network for binary classification tasks (e.g., predicting a $1$ or $0$, Yes or No, True or False).
* **How it works:** It takes the raw input and squashes it into a smooth 'S' shaped curve. A massive negative number gets squashed incredibly close to $0$, and a massive positive number gets squashed incredibly close to $1$. This transforms raw numbers into a clean percentage/probability.
* **Formula:** $f(x) = \frac{1}{1 + e^{-x}}$

**Tanh (Hyperbolic Tangent)**

* **What it is:** A mathematical function very similar to Sigmoid, but instead of squashing values between $0$ and $1$, it maps them to a range between $-1$ and $1$.
* **Its uses:** Used in hidden layers, particularly in Recurrent Neural Networks (RNNs) or in specific models where it is highly beneficial for the data to be centered around zero.
* **How it works:** It also creates an 'S' shaped curve, but because the center of the curve passes perfectly through the zero origin, negative inputs yield negative outputs and positive inputs yield positive outputs. This zero-centered nature often helps the network's weights update more evenly and efficiently during backpropagation compared to Sigmoid.
* **Formula:** $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Softmax**

* **What it is:** A function that takes a vector of raw, unnormalized numerical scores (logits) and converts them into a normalized probability distribution.
* **Its uses:** Used exclusively in the final output layer of multi-class classification networks (e.g., an image classifier deciding if a picture of a handwritten digit on the MNIST dataset is a 0 through 9).
* **How it works:** It applies an exponential function to each raw input score (which turns all negative numbers positive and amplifies the differences between them), and then divides each one by the sum of all the exponentiated scores. This forces every output to be a decimal between $0$ and $1$, and ensures that all the output values combined add up to exactly $1.0$ (representing 100% total probability across all categories).
* **Formula:** $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

> activations functions are connected to the concept vanishing gradients, which will be discussed later.
