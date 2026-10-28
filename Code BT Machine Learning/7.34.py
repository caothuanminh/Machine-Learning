import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
rng = np.random.default_rng(42)

image = np.zeros((120, 180, 3), dtype=np.uint8)
image[:60, :90] = [230, 90, 70]
image[:60, 90:] = [60, 170, 220]
image[60:, :90] = [240, 210, 80]
image[60:, 90:] = [50, 80, 60]

noise = rng.normal(0, 18, image.shape)
image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

pixels = image.reshape(-1, 3).astype(float)

model = KMeans(n_clusters=16, n_init=5, random_state=42)
labels = model.fit_predict(pixels)

compressed = model.cluster_centers_[labels].reshape(image.shape)
compressed = np.clip(compressed, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(compressed)
plt.title("Anh sau K-means, K=16")
plt.axis("off")

plt.tight_layout()
plt.savefig("kmeans_nen_anh.png", dpi=200)
plt.show()

print("So mau RGB goc:", len(np.unique(image.reshape(-1, 3), axis=0)))
print("So mau RGB sau nen:", len(np.unique(compressed.reshape(-1, 3), axis=0)))