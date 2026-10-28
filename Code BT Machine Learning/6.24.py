from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for C in [0.1, 1, 10]:
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    svm = model.named_steps["svc"]

    print(f"\nLinear SVM C={C}")
    print("Support vectors:", svm.n_support_.sum())
    print("Test accuracy:", accuracy_score(y_test, pred))

for C, gamma in [(0.1, "scale"), (1, "scale"), (10, "scale"), (1, 0.01), (10, 0.01)]:
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma)
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"RBF SVM C={C}, gamma={gamma}, accuracy={accuracy_score(y_test, pred):.4f}")