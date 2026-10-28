import numpy as np
import matplotlib.pyplot as plt

A = np.array([1.0, 1.0])
B = np.array([2.0, 2.0])
C = np.array([2.0, 0.0])

X = np.array([A, B, C])
y = np.array([1, 1, -1])

# Nghiem toi uu:
# w = (-2, 0), b = 3
# Sieu phang: -2*x1 + 3 = 0, hay x1 = 1.5
w = np.array([-2.0, 0.0])
b = 3.0

for xi, yi in zip(X, y):
    print("Diem:", xi)
    print("Score:", np.dot(w, xi) + b)
    print("y * score:", yi * (np.dot(w, xi) + b))

print("w:", w)
print("b:", b)
print("Margin mot ben:", 1 / np.linalg.norm(w))
print("Khoang cach hai bien:", 2 / np.linalg.norm(w))

plt.figure(figsize=(7, 6))
plt.scatter([A[0], B[0]], [A[1], B[1]], label="Lop +1")
plt.scatter([C[0]], [C[1]], label="Lop -1")

plt.axvline(1.5, label="Sieu phang")
plt.axvline(1.0, linestyle="--", label="Bien margin +1")
plt.axvline(2.0, linestyle="--", label="Bien margin -1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svm_ba_diem.png", dpi=200)
plt.show()