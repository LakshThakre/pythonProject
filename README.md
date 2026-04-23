# MNIST Neural Network from Scratch

This project demonstrates the internal mechanics of a neural network, including **forward propagation**, **backpropagation**, and **gradient descent**, without relying on high-level deep learning frameworks like TensorFlow or PyTorch for the training logic.

## 🚀 Features
* **Customizable Architecture**: Easily change the number of layers and neurons.
* **He Initialization**: Optimized weight initialization for ReLU activation functions.
* **ReLU & Softmax**: Implements rectified linear units for hidden layers and softmax for the output layer.
* **Backpropagation**: Manual implementation of the chain rule to calculate gradients.
* **Evaluation Tools**: Includes a confusion matrix and detailed classification report (Precision, Recall, F1-score).

---

## 🛠 Code Explanation

### 1. Initialization (`__init__`)
The network initializes weights using **He Initialization**, which scales weights based on the size of the previous layer:
$$W \approx N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$
This prevents gradients from vanishing or exploding early in training.

### 2. Forward Propagation (`forward`)
The data flows through the network using the following steps for each layer:
1.  **Linear Transformation**: $Z = A_{prev} \cdot W + b$
2.  **Activation**: 
    * **ReLU**: $f(z) = \max(0, z)$ for hidden layers.
    * **Softmax**: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$ for the output layer to produce probabilities.

### 3. Backpropagation (`backward`)
This is where the learning happens. The code calculates how much each weight contributed to the error using the chain rule. 
* It starts at the output layer by calculating the difference between predictions and actual values: `error = activations[-1] - y_true`.
* It then propagates this error backward to update hidden layers, multiplying by the `relu_derivative` to account for the activation function's slope.

### 4. Training Loop (`train`)
The training process uses **Stochastic Gradient Descent (SGD)** with mini-batches:
* **Shuffling**: Data is shuffled every epoch to ensure the model doesn't learn the order of samples.
* **Batching**: Processes small chunks (128 samples) at a time for efficiency.
* **Updating**: Weights are adjusted using the learning rate: $W = W - \eta \cdot \nabla W$.

### 5. Data Pipeline (`load_data_tf`)
While the model logic is NumPy-only, the script uses TensorFlow purely as a utility to download the MNIST dataset. It performs:
* **Flattening**: Converts $28 \times 28$ images into a 1D vector of $784$ pixels.
* **Normalization**: Scales pixel values from $[0, 255]$ to $[0, 1]$ to help the model converge faster.
* **One-Hot Encoding**: Converts labels (like `5`) into vectors (like `[0,0,0,0,0,1,0,0,0,0]`).

---

## 📊 Performance Metrics

After training, the model provides a **Confusion Matrix**. 



* **Precision**: Accuracy of positive predictions.
* **Recall**: Ability of the model to find all relevant cases (digits).
* **F1-Score**: The harmonic mean of Precision and Recall.

---

## 💻 Requirements
* `numpy`
* `tensorflow` (for data loading only)
* `scikit-learn` (for evaluation metrics)

## 🏃 How to Run
1. Ensure you have the dependencies installed: `pip install numpy tensorflow scikit-learn`.
2. Run the script: `python your_script_name.py`.
3. Watch the logs for epoch-by-epoch loss and validation accuracy updates!
