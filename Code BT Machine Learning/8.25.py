import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append((name, accuracy, training_time))

    print(f"Mô hình: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Thời gian huấn luyện: {training_time:.4f} giây")
    print("-" * 40)

names = [r[0] for r in results]
accuracies = [r[1] for r in results]
times = [r[2] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(names, accuracies)
axes[0].set_title("So sánh Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(names, times)
axes[1].set_title("So sánh thời gian huấn luyện")
axes[1].set_ylabel("Thời gian (giây)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("bai5_so_sanh_ensemble.png", dpi=300)
plt.show()

print("\nBẢNG TỔNG KẾT")
for name, accuracy, training_time in results:
    print(
        f"{name:20s} | "
        f"Accuracy = {accuracy:.4f} | "
        f"Time = {training_time:.4f}s"
    )