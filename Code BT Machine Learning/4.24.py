import numpy as np
def relu(x):
    return np.maximum(0, x)
x = np.array([[1.0], [2.0]])
W1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])
W2 = np.array([[1.0, 1.0]])
z1 = W1 @ x
a1 = relu(z1)
z2 = W2 @ a1
y_hat = relu(z2)
print("z^(1):\n", z1)
print("a^(1):\n", a1)
print("z^(2):\n", z2)
print("Dau ra cuoi cung:", y_hat)