import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n = len(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for _ in range(self.epochs):
            y_hat = self.sigmoid(X @ self.w + self.b)
            eps = 1e-12

            loss = -np.mean(
                y * np.log(y_hat + eps)
                + (1 - y) * np.log(1 - y_hat + eps)
            )
            self.loss_history.append(loss)

            dw = X.T @ (y_hat - y) / n
            db = np.mean(y_hat - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


iris = load_iris()
mask = iris.target < 2
X = iris.data[mask, :2]
y = iris.target[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.figure(figsize=(9, 5))

for lr in [0.01, 0.1, 1.0]:
    model = LogisticRegressionGD(lr, 1000)
    model.fit(X_train, y_train)

    accuracy = np.mean(model.predict(X_test) == y_test)
    print("Learning rate:", lr)
    print("Accuracy:", accuracy)
    print("Final loss:", model.loss_history[-1])

    plt.plot(model.loss_history, label=f"lr={lr}")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logistic_loss.png", dpi=200)
plt.show()