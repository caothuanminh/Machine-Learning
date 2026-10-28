import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) == 1, 1, -1)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(self.w, xi) + self.b >= 0 else -1
                if pred != yi:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi
                    errors += 1
            if errors == 0:
                break
        return self
    def predict(self, X):
        scores = np.asarray(X) @ self.w + self.b
        return np.where(scores >= 0, 1, -1)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)
y = np.where(y == 1, 1, -1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Perceptron(learning_rate=0.01, epochs=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label=1, zero_division=0))
print("Recall:", recall_score(y_test, pred, pos_label=1, zero_division=0))
print("F1-score:", f1_score(y_test, pred, pos_label=1, zero_division=0))