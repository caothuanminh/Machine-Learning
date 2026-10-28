import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = 1 if np.dot(self.w, xi) + self.b >= 0 else -1
                if prediction != yi:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi
                    errors += 1
            if errors == 0:
                break
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)
if __name__ == "__main__":
    X = np.array([
        [2, 2],
        [1, 1],
        [-1, -1],
        [-2, -2]
    ])
    y = np.array([1, 1, -1, -1])
    model = Perceptron(learning_rate=0.1, epochs=100)
    model.fit(X, y)
    print("Trong so:", model.w)
    print("Bias:", model.b)
    print("Du doan:", model.predict(X))
    print("Nhan thuc te:", y)