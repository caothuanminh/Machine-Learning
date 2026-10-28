import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)

X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = 2 * X.ravel() + np.sin(X.ravel()) + rng.normal(0, 0.4, 50)

X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X.ravel(), y, label="Du lieu goc")

for depth in [2, 5, 10]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X, y)

    pred = model.predict(X)
    mse = mean_squared_error(y, pred)

    plt.plot(
        X_plot.ravel(),
        model.predict(X_plot),
        label=f"depth={depth}, MSE={mse:.3f}"
    )

    print(f"max_depth={depth}, MSE train={mse:.4f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("CART Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/bai_5_24_cart_regression.png", dpi=200)
plt.show()