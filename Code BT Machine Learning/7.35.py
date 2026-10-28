import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(
    n_samples=450,
    centers=[[-5, 0], [0, 5], [5, 0]],
    cluster_std=[1.7, 0.35, 0.8],
    random_state=42
)

random_inertias = []
plus_inertias = []

for seed in range(20):
    model_random = KMeans(
        n_clusters=3,
        init="random",
        n_init=1,
        random_state=seed
    )
    model_random.fit(X)
    random_inertias.append(model_random.inertia_)

    model_plus = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=1,
        random_state=seed
    )
    model_plus.fit(X)
    plus_inertias.append(model_plus.inertia_)

print("Random - trung binh:", np.mean(random_inertias))
print("Random - min:", np.min(random_inertias))
print("Random - max:", np.max(random_inertias))

print("K-means++ - trung binh:", np.mean(plus_inertias))
print("K-means++ - min:", np.min(plus_inertias))
print("K-means++ - max:", np.max(plus_inertias))

plt.figure(figsize=(9, 5))
plt.hist(random_inertias, bins=10, alpha=0.6, label="Random")
plt.hist(plus_inertias, bins=10, alpha=0.6, label="K-means++")
plt.xlabel("Inertia")
plt.ylabel("So lan")
plt.title("So sanh khoi tao K-means")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_so_sanh_khoi_tao.png", dpi=200)
plt.show()