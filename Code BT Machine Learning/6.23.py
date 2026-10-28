import numpy as np
import matplotlib.pyplot as plt
X_pos = np.array([[0, 0], [1, 0]])
X_neg = np.array([[2, 1], [3, 1]])

X = np.vstack([X_pos, X_neg])
y = np.array([1, 1, -1, -1])

w = np.array([-1.0, -1.0])
b = 1.5

# Nghiem hard-margin toi uu cho bo du lieu nay:
# w = (-1, -1), b = 1.5
# Hai diem (1,0) va (2,1) nam tren bien margin.

for xi, yi in zip(X, y):
    print(xi, "score =", np.dot(w, xi) + b)

support_vectors = []
for xi, yi in zip(X, y):
    if np.isclose(yi * (np.dot(w, xi) + b), 0.5):
        support_vectors.append(xi)

print("w:", w)
print("b:", b)
print("Support vectors:", support_vectors)
print("Margin mot ben:", 0.5 / np.linalg.norm(w))
print("Khoang cach hai bien:", 1 / np.linalg.norm(w))

plt.figure(figsize=(8, 6))
plt.scatter(X_pos[:, 0], X_pos[:, 1], label="Lop +1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], label="Lop -1")

x1 = np.linspace(-1, 4, 100)
plt.plot(x1, 1.5 - x1, label="Sieu phang phan cach")
plt.plot(x1, 1.0 - x1, "--", label="Bien margin")
plt.plot(x1, 2.0 - x1, "--")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_hard_margin.png", dpi=200)
plt.show()