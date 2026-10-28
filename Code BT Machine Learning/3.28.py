import numpy as np
w = np.array([-2.0, 1.0, 0.0])
x = np.array([2.0, 3.0, 1.0])
y = 1
learning_rate = 1.0
z = np.dot(w, x)
y_pred = 1 if z >= 0 else -1
print("Truoc cap nhat:")
print("w^T x =", z)
print("Nhan du doan:", y_pred)
if y_pred != y:
    w = w + learning_rate * y * x
print("Trong so sau cap nhat:", w)
print("w^T x sau cap nhat:", np.dot(w, x))