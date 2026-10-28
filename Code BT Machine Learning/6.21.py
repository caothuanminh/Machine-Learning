import numpy as np
X = np.array([1.0, 2.0, 4.0])
y = np.array([0.0, 0.0, 1.0])

w = 0.5
b = -1.5
lr = 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = w * X + b
y_hat = sigmoid(z)

print("Xac suat du doan:")
for xi, yi, zi, pi in zip(X, y, z, y_hat):
    print(f"x={xi}, y={yi}, z={zi:.6f}, y_hat={pi:.6f}")

eps = 1e-12
loss = -np.mean(
    y * np.log(y_hat + eps)
    + (1 - y) * np.log(1 - y_hat + eps)
)

dw = np.mean((y_hat - y) * X)
db = np.mean(y_hat - y)

print("Cross-Entropy:", loss)
print("dw:", dw)
print("db:", db)

w -= lr * dw
b -= lr * db

print("w sau cap nhat:", w)
print("b sau cap nhat:", b)