import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


def run_dbscan(X, eps, min_samples):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    labels = model.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)

    return labels, n_clusters, n_noise


if __name__ == "__main__":
    X, y_true = make_circles(
        n_samples=500,
        factor=0.5,
        noise=0.05,
        random_state=42
    )

    params = [
        (0.2, 5),
        (0.5, 5),
        (1.0, 5),
        (0.5, 10)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (eps, min_samples) in enumerate(params):
        labels, n_clusters, n_noise = run_dbscan(
            X, eps, min_samples
        )

        ari = adjusted_rand_score(y_true, labels)

        print("=" * 40)
        print(f"eps = {eps}, min_samples = {min_samples}")
        print("Số cụm:", n_clusters)
        print("Số điểm nhiễu:", n_noise)
        print("Adjusted Rand Index:", round(ari, 4))

        axes[i].scatter(
            X[:, 0],
            X[:, 1],
            c=labels,
            cmap="viridis",
            s=25
        )
        axes[i].set_title(
            f"eps={eps}, min_samples={min_samples}\n"
            f"Cụm={n_clusters}, nhiễu={n_noise}, ARI={ari:.3f}"
        )
        axes[i].set_xlabel("X1")
        axes[i].set_ylabel("X2")

    plt.tight_layout()
    plt.savefig("bai8_dbscan_tham_so.png", dpi=300)
    plt.show()