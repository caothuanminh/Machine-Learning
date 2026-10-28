import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import rand_score

class MyKMeans:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, K, max_iters=100):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(len(X), K, replace=False)
        centers = X[indices].copy()

        for _ in range(max_iters):
            distances = np.sum(
                (X[:, None, :] - centers[None, :, :]) ** 2,
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()

            for k in range(K):
                points = X[labels == k]
                if len(points) > 0:
                    new_centers[k] = points.mean(axis=0)

            if np.allclose(new_centers, centers):
                centers = new_centers
                break

            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.sum(
            (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2
        )
        return np.argmin(distances, axis=1)


iris = load_iris()
X = iris.data[:, :2]
y_true = iris.target

model = MyKMeans(random_state=42).fit(X, K=3)

print("Tam cum:")
print(model.cluster_centers_)
print("Rand Index:", rand_score(y_true, model.labels_))

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Tam cum"
)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_iris.png", dpi=200)
plt.show()