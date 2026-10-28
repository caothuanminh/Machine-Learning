import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

os.makedirs("outputs", exist_ok=True)

for depth in [None, 2, 3]:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("max_depth:", depth)
    print("Train accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))

    if depth is None:
        plt.figure(figsize=(15, 9))
        plot_tree(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True,
            rounded=True
        )
        plt.tight_layout()
        plt.savefig("outputs/bai_5_23_cay_iris.png", dpi=200)
        plt.close()