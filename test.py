import numpy as np
import time
from numba import jit
from tqdm import tqdm

# Sigmoid and derivative
@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@jit(nopython=True)
def sigmoid_derivative(y):
    return y * (1 - y)

@jit(nopython=True)
def random_nb_gen():
    return (2 * np.random.rand()) - 1  # match C's [-1, 1] style range


# XOR-like dataset (same as in your C code)
inputs = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

outputs = np.array([[1.0], [0.0], [0.0], [1.0]])

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.rand(2, 2)
b1 = np.random.rand(1, 2)
w2 = np.random.rand(2, 1)
b2 = np.random.rand(1, 1)

lr = 0.5
epochs = 2300062

start_time = time.time()

# Train the network with progress bar and numba-optimized functions
for epoch in tqdm(range(epochs), desc="Training"):
    # Forward pass
    h = sigmoid(inputs @ w1 + b1)
    o = sigmoid(h @ w2 + b2)

    # Backpropagation
    error = outputs - o
    delta_o = error * sigmoid_derivative(o)
    delta_h = delta_o @ w2.T * sigmoid_derivative(h)

    # Gradient accumulation
    dw2 = h.T @ delta_o
    db2 = np.sum(delta_o, axis=0, keepdims=True)
    dw1 = inputs.T @ delta_h
    db1 = np.sum(delta_h, axis=0, keepdims=True)

    # Update weights and biases
    w2 += lr * dw2
    b2 += lr * db2
    w1 += lr * dw1
    b1 += lr * db1
    loss = np.mean(error ** 2)
    if epoch % 10000 == 0:
        print(f"Loss: {loss:.6f}")
execution_time = (time.time() - start_time) * 1000
print(f"Execution time: {execution_time:.6f} ms")

# Evaluation
h = sigmoid(np.dot(inputs, w1) + b1)
o = sigmoid(np.dot(h, w2) + b2)
rounded = np.round(o)

for i in range(4):
    print(f"Input: {inputs[i]} -> Output: {int(rounded[i][0])} "
          f"(Output before rounding: {o[i][0]:.6f})")

correct = np.sum(rounded.flatten() == outputs.flatten())
mse = np.mean((outputs - o) ** 2)

print(f"Accuracy: {correct}/4 ({(correct/4)*100:.1f}%)")
print(f"MSE: {mse:.6f}")

print("\nHidden Layer Weights:")
print(w1)
print("Hidden Layer Biases:")
print(b1)
print("Output Layer Weights:")
print(w2)
print("Output Layer Bias:")
print(b2)
