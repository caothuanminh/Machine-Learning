import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error

os.makedirs("outputs", exist_ok=True)

csv_file = "khach_hang.csv"

if not os.path.exists(csv_file):
    rng = np.random.default_rng(42)
    n = 200
    age = rng.integers(18, 66, n)
    income = rng.integers(5, 51, n)
    visits = rng.integers(0, 21, n)
    spending = income * visits + rng.normal(0, 30, n)
    buy = ((income > 20) & (visits > 5)).astype(int)

    pd.DataFrame({
        "Tuoi": age,
        "ThuNhap": income,
        "SoLanTruyCap": visits,
        "ChiTieu": spending,
        "MuaHang": buy
    }).to_csv(csv_file, index=False)

df = pd.read_csv(csv_file)

X = df[["Tuoi", "ThuNhap", "SoLanTruyCap"]]
y_class = df["MuaHang"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.25, random_state=42, stratify=y_class
)

for leaf in [1, 5, 10]:
    model = DecisionTreeClassifier(
        criterion="gini",
        min_samples_leaf=leaf,
        random_state=42
    )
    model.fit(X_train, y_train)

    print(f"\nmin_samples_leaf={leaf}")
    print("Train accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))

    if leaf == 5:
        plt.figure(figsize=(14, 8))
        plot_tree(
            model,
            feature_names=X.columns,
            class_names=["Khong mua", "Mua"],
            filled=True,
            rounded=True
        )
        plt.tight_layout()
        plt.savefig("outputs/bai_5_25_cay_phan_lop.png", dpi=200)
        plt.close()

print("\nFeature importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(name, round(importance, 4))

if "ChiTieu" in df.columns:
    y_reg = df["ChiTieu"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )

    reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    reg.fit(X_train_r, y_train_r)

    pred = reg.predict(X_test_r)
    print("\nRegression MSE:", mean_squared_error(y_test_r, pred))

    plt.figure(figsize=(14, 8))
    plot_tree(reg, feature_names=X.columns, filled=True, rounded=True)
    plt.tight_layout()
    plt.savefig("outputs/bai_5_25_cay_hoi_quy.png", dpi=200)
    plt.close()