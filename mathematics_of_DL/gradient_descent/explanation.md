We know about gradient descent, and lets just see the different gradient descent variants and how do they work in DL and also clear some doubts.

I recommend you to look out this [gradient descent script](../gradient_descent/gradient_descent.ipynb) to initially understand how scratch code looks. The file also has raw `tensorflow` code too, but just watch the scratch code.

There are mainly three variants:
1) *Batch GD*
2) *Stochastic GD (SGD)*
3) *Mini Batch GD* 

### The script uses **Batch GD** and here is the proof:
In this scratch code, two lines prove it is Batch Gradient Descent:
1. No inner loop over samples: Inside for i in range(epochs):, there is only one loop (the epoch loop). You pass the entire 22-row age and affordability columns into weighted_sum at once.
2. Averaging over n (len(age)) before updating weights:
- np.dot(age, error) automatically sums the element-wise products across all $n = 22$ samples.
- Dividing by n (1/n) computes the average gradient across the entire training dataset before executing a single weight update (w1 = w1 - rate * w1d).
> (Notice also in our Keras output: 1/1 ━━━━━━━━━━━━━━━━━━━━ — Keras printed 1/1 batch per epoch because the default batch size is 32 and you had 22 samples, meaning Keras also ran pure Batch GD!)

The uploaded file is truncated right inside the Keras `model.fit()` output, so the custom gradient descent function at the bottom isn't visible. However, based on the exact codebasics tutorial (`gradient_descent.ipynb` for insurance data) and your reference to `np.dot(np.transpose(...))`, here is the exact breakdown of your scratch implementation.

#### 1. Why `np.dot(np.transpose(age), (y_predicted - y_true))` is Used (and How to Simplify It)

In the tutorial, the derivative of Log Loss with respect to $w_1$ (age) is:

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{n} \sum_{i=1}^{n} \text{age}_i \cdot (\hat{y}_i - y_i)$$

To compute $\sum (\text{age}_i \times \text{error}_i)$ in one shot without a `for` loop, you take the dot product of the two vectors:

* `age` is a 1D array of shape `(22,)` (22 training samples).


* `(y_predicted - y_true)` is also a 1D array of shape `(22,)`.

**Why the tutorial uses `np.transpose()`:**

In strict 2D matrix math, to multiply two column vectors of shape `(22, 1)`, you must transpose the first one into a row vector `(1, 22)` so the inner dimensions match: $(1 \times \mathbf{22}) \cdot (\mathbf{22} \times 1) \rightarrow (1 \times 1)$.

**How to simplify it:**

Because NumPy 1D arrays `(22,)` have no row/column orientation, `np.transpose()` on a 1D array does nothing at all. You can replace `np.dot(np.transpose(age), (y_predicted - y_true))` with a clean `@` operator or direct `np.dot`:

```python
# Original verbose way:
w1d = (1/n) * np.dot(np.transpose(age), (y_predicted - y_true))

# Simplified Pythonic way:
error = y_predicted - y_true
w1d = (1/n) * (age @ error)
w2d = (1/n) * (affordability @ error)
bd  = np.mean(error)  # Replaces (1/n) * np.sum(error)

```

#### 2. How the Code Differs Across All 3 Gradient Descent Types

| Type | Samples per Weight Update | Updates per Epoch (for 22 samples) | Key Code Signature |
| --- | --- | --- | --- |
| **Batch GD** | All $N$ samples (`22`) | **1 update** | Vectorized dot product over full arrays, divided by `n` |
| **Stochastic GD (SGD)** | **1 random sample** | **22 updates** | Nested loop picking 1 row at a time (`random_index`), no `1/n` |
| **Mini-Batch GD** | Subset $B$ (e.g., `5` samples) | **$\lceil 22/5 \rceil = 5$ updates** | Nested loop slicing arrays `[j : j+batch_size]`, divided by `batch_size` |

#### Side-by-Side Code Comparison

```python
# ==========================================
# 1. BATCH GRADIENT DESCENT (Your Current Code)
# ==========================================
for i in range(epochs):
    # Forward pass on ALL 22 samples at once
    weighted_sum = w1 * age + w2 * affordability + bias
    y_predicted = sigmoid_numpy(weighted_sum)
    
    # Gradient averaged over ALL n samples
    error = y_predicted - y_true
    w1d = (1/n) * (age @ error)
    w2d = (1/n) * (affordability @ error)
    bd  = np.mean(error)
    
    # Weights updated ONCE per epoch
    w1 -= rate * w1d
    w2 -= rate * w2d
    bias -= rate * bd


# ==========================================
# 2. STOCHASTIC GRADIENT DESCENT (SGD)
# ==========================================
for i in range(epochs):
    # Inner loop runs once for EVERY single sample
    for j in range(n):
        idx = np.random.randint(0, n) # Pick 1 random sample
        x1, x2, y = age[idx], affordability[idx], y_true[idx]
        
        y_predicted = sigmoid_numpy(w1 * x1 + w2 * x2 + bias)
        
        # Gradient from just THAT ONE sample (no 1/n average)
        error = y_predicted - y
        w1d = x1 * error
        w2d = x2 * error
        bd  = error
        
        # Weights updated 22 TIMES per epoch
        w1 -= rate * w1d
        w2 -= rate * w2d
        bias -= rate * bd


# ==========================================
# 3. MINI-BATCH GRADIENT DESCENT
# ==========================================
batch_size = 5
for i in range(epochs):
    # Shuffle data each epoch
    indices = np.random.permutation(n)
    age_s, aff_s, y_s = age[indices], affordability[indices], y_true[indices]
    
    # Step through dataset in chunks of batch_size
    for j in range(0, n, batch_size):
        x1_b = age_s[j : j + batch_size]
        x2_b = aff_s[j : j + batch_size]
        y_b  = y_s[j : j + batch_size]
        
        y_predicted = sigmoid_numpy(w1 * x1_b + w2 * x2_b + bias)
        
        # Gradient averaged over ONLY the current mini-batch
        error = y_predicted - y_b
        w1d = (1/len(x1_b)) * (x1_b @ error)
        w2d = (1/len(x2_b)) * (x2_b @ error)
        bd  = np.mean(error)
        
        # Weights updated 5 TIMES per epoch
        w1 -= rate * w1d
        w2 -= rate * w2d
        bias -= rate * bd

```

---

### Summary 
* **Batch GD:** Computes `error` over the entire array $\rightarrow$ updates weights **once** at the end of the epoch. Smooth convergence, slow on massive datasets.
* **SGD:** Updates weights **inside** a per-sample loop $\rightarrow$ noisy/zig-zag path, fast updates, escapes local minima.
* **Mini-Batch GD:** Slices arrays into chunks of `batch_size` $\rightarrow$ standard used in modern deep learning (PyTorch/Keras `batch_size=32`), balancing GPU matrix speed with frequent updates.

---

## Doubt related to how does input differ in a NN based one different variants
The key detail to understand first is that **the Input Layer shape never changes** -only the **number of rows (batch dimension)** fed into that input layer per forward/backward pass changes.

Let's use a concrete neural network example to trace the exact matrix shapes entering the input layer.

### Example Setup: House Price Prediction Neural Network

* **Dataset:** **100 houses** ($N = 100$).
* **Features per house:** **3 features** (`Size`, `Bedrooms`, `Age`).
* **Input Layer Definition:** `keras.Input(shape=(3,))` (expects 3 features per sample).
* **Hidden Layer:** 4 neurons $\rightarrow$ Weight matrix $W_1$ has shape `(3, 4)`.

No matter which Gradient Descent variant we use, the model's weight matrix $W_1$ is always `(3, 4)`. What changes is the **first dimension (number of rows)** of the input matrix $X$ passed into the network at each step.

### How Data Enters the Input Layer Across All 3 Variants

#### 1. Batch Gradient Descent (BGD)

You pass the **entire dataset** into the input layer at once.

* **Input Matrix Shape Entering Input Layer:** `(100, 3)`
* **Forward Pass Matrix Math:** `(100, 3)` $\times$ `(3, 4)` = `(100, 4)`
* **Steps per Epoch:** **1 step** (all 100 houses are processed in a single matrix multiplication).
* **Weight Updates per Epoch:** **1 update** after averaging the error across all 100 houses.

```text
[ All 100 Houses: (100, 3) ] ──► [ Input Layer (shape=(3,)) ] ──► 1 Weight Update

```

#### 2. Stochastic Gradient Descent (SGD)

You pass **1 single house** into the input layer at a time.

* **Input Matrix Shape Entering Input Layer:** `(1, 3)`
* **Forward Pass Matrix Math:** `(1, 3)` $\times$ `(3, 4)` = `(1, 4)`
* **Steps per Epoch:** **100 steps** (a `for` loop feeds 1 row at a time).
* **Weight Updates per Epoch:** **100 updates** (weights are updated immediately after each individual house).

```text
[ House 1: (1, 3)   ] ──► [ Input Layer (shape=(3,)) ] ──► Weight Update 1
[ House 2: (1, 3)   ] ──► [ Input Layer (shape=(3,)) ] ──► Weight Update 2
       ...                                                      ...
[ House 100: (1, 3) ] ──► [ Input Layer (shape=(3,)) ] ──► Weight Update 100

```

#### 3. Mini-Batch Gradient Descent

You slice the 100 houses into smaller chunks—say **`batch_size = 20`**—and pass one chunk at a time.

* **Input Matrix Shape Entering Input Layer:** `(20, 3)`
* **Forward Pass Matrix Math:** `(20, 3)` $\times$ `(3, 4)` = `(20, 4)`
* **Steps per Epoch:** **5 steps** ($100 \div 20 = 5$ batches).
* **Weight Updates per Epoch:** **5 updates** (weights are updated once after each batch of 20 houses).

```text
[ Batch 1 (Houses 1-20):   (20, 3) ] ──► [ Input Layer ] ──► Weight Update 1
[ Batch 2 (Houses 21-40):  (20, 3) ] ──► [ Input Layer ] ──► Weight Update 2
[ Batch 3 (Houses 41-60):  (20, 3) ] ──► [ Input Layer ] ──► Weight Update 3
[ Batch 4 (Houses 61-80):  (20, 3) ] ──► [ Input Layer ] ──► Weight Update 4
[ Batch 5 (Houses 81-100): (20, 3) ] ──► [ Input Layer ] ──► Weight Update 5

```

### How This Looks in Keras `model.fit()`

In Keras, you do not need to rewrite your network or change `layers.Input(shape=(3,))`. You control which GD variant is used strictly through the `batch_size` argument in `.fit()`:

```python
# 1. Batch Gradient Descent (passes all 100 rows at once)
model.fit(X_train, y_train, batch_size=100, epochs=50)

# 2. Stochastic Gradient Descent (passes 1 row at a time)
model.fit(X_train, y_train, batch_size=1, epochs=50)

# 3. Mini-Batch Gradient Descent (passes 20 rows at a time; Keras default is 32)
model.fit(X_train, y_train, batch_size=20, epochs=50)

```

### Side-by-Side Summary ($N = 100$ Houses, $3$ Features)

| GD Type | Shape Fed to Input Layer | Batches / Steps per Epoch | Weight Updates per Epoch |
| --- | --- | --- | --- |
| **Batch GD** | `(100, 3)` | 1 | 1 |
| **Stochastic GD (SGD)** | `(1, 3)` | 100 | 100 |
| **Mini-Batch GD (`batch_size=20`)** | `(20, 3)` | 5 | 5 |

*(Notice why Keras displays `shape=(None, 3)` when we run `model.summary()`: that `None` placeholder is specifically reserved for the variable batch size—whether it receives `100`, `1`, or `20` rows at runtime).*