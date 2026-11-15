If you see the predicting numbers problem we solved in the start, you can see that we simply divide the train and test set 255, because colors only range between 0 and 255, and here's why and info about other scalars.

### 1. Why He Used `X / 255` Instead of a Scikit-Learn Scaler

In the MNIST tutorial, dividing by `255` is mathematically identical to **Min-Max Scaling**, but done manually for three specific reasons:

1. **Known, Fixed Boundaries:** Every 8-bit grayscale image pixel is strictly bounded between `0` (pure black) and `255` (pure white). Because the minimum (`0`) and maximum (`255`) are universal constants rather than unknown statistics, `MinMaxScaler()` would literally compute `(X - 0) / (255 - 0)`, which simplifies directly to `X / 255`.
2. **Preserves Spatial & Background Zeros:** In MNIST, `0` represents the empty black background around the digit. Dividing by `255` keeps `0` as `0.0` (sparse data, which ReLU activations and matrix multiplications process cleanly). A `StandardScaler` would shift the black background to a negative number (around `-0.42`), forcing neurons to compute non-zero activations on empty space.
3. **No 3D-to-2D Reshaping Overhead:** `keras.datasets.mnist` loads `X_train` as a 3D array of shape `(60000, 28, 28)`. Scikit-learn scalers only accept 2D arrays `(samples, features)`, so using `MinMaxScaler` would force you to flatten the images before scaling and reshape them back afterward. Dividing by `255` works natively on 3D or 4D image tensors.

---

### 2. Main Types of Scalers & When to Use Each

When working with tabular datasets or features without fixed `0–255` pixel bounds, you choose a scaler based on the **distribution** and **outlier behavior** of your data:

| Scaler | Formula | Output Range / Properties | When to Use | Example Dataset |
|---|---|---|---|---|
| **Min-Max Scaler** | $$X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$ | Strictly bounded between **`[0, 1]`** (or `[-1, 1]` when configured). | 1. Data has known fixed bounds.<br>2. Data is not necessarily normally distributed.<br>3. Dataset has no extreme outliers.<br>4. Image pixel intensities (`0–255`). | **MNIST / Image Datasets** (`0–255` pixels) or **Sensor Readings** with fixed hardware limits (e.g., humidity `0–100%`). |
| **Standard Scaler (Z-Score)** | $$X' = \frac{X - \mu}{\sigma}$$ | Mean = **`0`**, Standard Deviation = **`1`** (unbounded). | 1. Features are approximately Gaussian (normally distributed).<br>2. Default choice for linear models, MLPs, SVMs, and PCA.<br>3. Outliers are mild. | **Medical / Biometric Data** (e.g., patient height, weight, blood pressure) or **Exam Scores**. |
| **Robust Scaler** | $$X' = \frac{X - Q_{50}}{Q_{75} - Q_{25}}$$ | Centered at median **`0`**, scaled by Interquartile Range (IQR). | 1. Dataset contains heavy outliers.<br>2. Outliers would otherwise distort scaling using Min-Max or Standard Scaler. | **Real Estate / House Prices** (where a few $50M mansions skew the average) or **Personal Income / Fraud Detection** datasets. |
| **MaxAbs Scaler** | $$X' = \frac{X}{|X_{\max}|}$$ | Values bounded between **`[-1, 1]`** (assuming the maximum absolute value is used). | 1. Data is already centered at zero or is sparse (contains mostly `0`s).<br>2. You want to preserve exact zero values. | **TF-IDF / Bag-of-Words Text Matrices** (NLP datasets where most word counts in a document are `0`). |
| **Log Transformation** *(Power / Nonlinear)* | $$X' = \log(X + 1)$$ | Compresses large right-skewed values into a tighter range. | 1. Data is heavily right-skewed.<br>2. Values span multiple orders of magnitude.<br>3. Often applied before StandardScaler. | **Website Traffic / Clicks**, **City Populations**, or **E-commerce Revenue**. |

---

### 3. Critical Rule for Deep Learning Pipelines: Prevent Data Leakage

Whenever you use a stateful scaler (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) in a real ML/DL workflow, **always fit only on the training set**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. Calculate mean & std ONLY from X_train, then scale X_train
X_train_scaled = scaler.fit_transform(X_train)

# 2. Apply X_train's mean & std to X_test (NEVER call fit_transform on X_test!)
X_test_scaled = scaler.transform(X_test)

```

If you call `fit_transform` on `X_test`, your model peeks at the test set's distribution during preprocessing (**data leakage**), which invalidates your test accuracy. *(With `X / 255` on MNIST, leakage isn't an issue because `255` is a universal constant rather than a statistic learned from the data).*