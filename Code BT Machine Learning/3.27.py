import numpy as np
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
z = np.dot(w, x)
y_pred = 1 if z >= 0 else -1
y_true = -1
print("w^T x =", z)
print("Nhan du doan:", y_pred)
print("Nhan thuc te:", y_true)
print("Phan lop sai:", y_pred != y_true)