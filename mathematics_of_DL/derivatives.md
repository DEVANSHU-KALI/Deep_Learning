# Derivatives and partial derivatives
to simply understand derivatives, lest see from where did this concepts arrive.

**The Foundation: Slopes and Regular Derivatives**

* **Straight Lines:** To find the rate of change on a straight line, you use the standard slope formula $\frac{\Delta y}{\Delta x}$.
* **Curved Lines (Regular Derivatives):** For non-linear curves, there is no single constant slope. You must "zoom in" infinitely close to a single point until the curve appears flat, and calculate the slope exactly at that microscopic point. This is a **derivative** (like $\frac{d}{dx} x^2 = 2x$). It tells you exactly how much an output ($y$) changes when you tweak a single input ($x$).

**The Problem: Why Partial Derivatives Entered the Math**
Regular derivatives only work in a strictly 2D world where your outcome depends on exactly *one* variable (e.g., $y = f(x)$).

But the real world, and Deep Learning, is multi-dimensional. If you are calculating the price of a house ($z$) based on both Square Footage ($x$) and Age ($y$), a regular derivative mathematically crashes. It cannot answer "how does the price change?" because it doesn't know if you are changing the size, the age, or both.

**What They Tell Us:** Partial derivatives ($\partial$) solve this by isolating variables in a multi-dimensional space. They tell you the rate of change for *one specific variable* while freezing every other variable in place, treating them as fixed numbers.

**A Deep Learning Example Where Regular Derivatives Fail**
Imagine a tiny neural network calculating its Error (Loss) using two weights: $\text{Loss} = (w_1 + w_2)^2$.

If you try to use a regular derivative to find the slope of the Loss, it is impossible. If you tweak $w_1$, the Loss changes. If you tweak $w_2$, the Loss also changes. The curve exists in 3D space, like a bowl.

Partial derivatives solve this. During training, the optimizer calculates $\frac{\partial \text{Loss}}{\partial w_1}$. It temporarily pretends $w_2$ is just a static number (like the number 5). This isolates $w_1$, revealing exactly how much $w_1$ *alone* contributed to the error. It then repeats the process to find $\frac{\partial \text{Loss}}{\partial w_2}$.

**Where They Are Used in Deep Learning**

* **Regular Derivatives are used in Activation Functions:** Functions like ReLU or Sigmoid only take one number in and spit one number out ($y = \text{ReLU}(x)$). When backpropagation passes through an activation layer, it uses a standard, single-variable derivative to find the slope of the curve at that exact point.
* **Partial Derivatives are the engine of Backpropagation (Gradient Descent):** Modern neural networks do not have one variable; they have millions of weights and biases. The word **"Gradient"** simply means a giant list of partial derivatives. During the backward pass, PyTorch calculates a partial derivative for *every single parameter in the network*, figuring out exactly how to adjust one specific weight while holding the other billions of weights mathematically frozen.
