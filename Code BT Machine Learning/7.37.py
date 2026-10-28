import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, X):
        X = np.asarray(X)
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        distances = np.sqrt(
            np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        )

        def region_query(point_idx):
            return np.where(distances[point_idx] <= self.eps)[0]

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = region_query(i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = list(neighbors)
            in_seeds = set(seeds)
            j = 0

            while j < len(seeds):
                point = seeds[j]

                if not visited[point]:
                    visited[point] = True
                    point_neighbors = region_query(point)

                    if len(point_neighbors) >= self.min_samples:
                        for neighbor in point_neighbors:
                            if neighbor not in in_seeds:
                                seeds.append(neighbor)
                                in_seeds.add(neighbor)

                if labels[point] == -1:
                    labels[point] = cluster_id

                j += 1

            cluster_id += 1

        self.labels_ = labels
        return labels


if __name__ == "__main__":
    X, y_true = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=0.6,
        random_state=42
    )

    rng = np.random.default_rng(42)
    noise = rng.uniform(-8, 8, size=(50, 2))
    X = np.vstack([X, noise])

    eps = 0.5
    min_samples = 5

    my_dbscan = MyDBSCAN(eps=eps, min_samples=min_samples)
    labels_custom = my_dbscan.fit_predict(X)

    sklearn_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_sklearn = sklearn_dbscan.fit_predict(X)

    print("===== DBSCAN THỦ CÔNG =====")
    print("Số cụm:", len(set(labels_custom) - {-1}))
    print("Số điểm nhiễu:", np.sum(labels_custom == -1))

    print("\n===== DBSCAN SCIKIT-LEARN =====")
    print("Số cụm:", len(set(labels_sklearn) - {-1}))
    print("Số điểm nhiễu:", np.sum(labels_sklearn == -1))

    print(
        "\nAdjusted Rand Index:",
        round(adjusted_rand_score(labels_custom, labels_sklearn), 4)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        X[:, 0], X[:, 1],
        c=labels_custom,
        cmap="viridis",
        s=25
    )
    axes[0].set_title("DBSCAN thủ công")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    axes[1].scatter(
        X[:, 0], X[:, 1],
        c=labels_sklearn,
        cmap="viridis",
        s=25
    )
    axes[1].set_title("DBSCAN scikit-learn")
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")

    plt.tight_layout()
    plt.savefig("bai7_dbscan_so_sanh.png", dpi=300)
    plt.show()