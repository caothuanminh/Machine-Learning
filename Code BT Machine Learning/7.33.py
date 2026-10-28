import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y_true = make_blobs(
    n_samples=500,
    centers=5,
    cluster_std=0.7,
    random_state=42
)

inertias = []

for K in range(1, 11):
    model = KMeans(n_clusters=K, n_init=10, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
    print(f"K={K}, inertia={model.inertia_:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=200)
plt.show()

model = KMeans(n_clusters=5, n_init=10, random_state=42)
labels = model.fit_predict(X)
print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))