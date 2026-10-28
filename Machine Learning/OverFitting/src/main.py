import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 Bắt đầu phân tích Overfitting trong Dự báo Giá nhà...\n")

df = pd.read_csv('data/train.csv')

target_col = 'Price'  
if target_col not in df.columns:
    possible_targets = ['Gia', 'Target', 'SalePrice', 'price', 'GiaNha']
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

print(f"✅ Đã tìm thấy cột mục tiêu (Target): '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col]

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*50)
print("🔴 PHẦN 1: MÔ HÌNH BỊ OVERFITTING")
print("="*50)

model_overfit = DecisionTreeRegressor(random_state=42)
model_overfit.fit(X_train, y_train)

y_train_pred_over = model_overfit.predict(X_train)
y_test_pred_over = model_overfit.predict(X_test)

r2_train_over = r2_score(y_train, y_train_pred_over)
r2_test_over = r2_score(y_test, y_test_pred_over)

print(f"R2 Score trên tập Train: {r2_train_over:.4f} (Gần bằng 1.0 là dấu hiệu học thuộc lòng)")
print(f"R2 Score trên tập Test:  {r2_test_over:.4f} (Thấp hơn rất nhiều so với tập Train)")
print(f"⚠️ Khoảng cách R2 (Gap): {r2_train_over - r2_test_over:.4f} (Gap càng lớn, Overfitting càng nặng)")

print("\n" + "="*50)
print("🟢 PHẦN 2: MÔ HÌNH ĐÃ KHẮC PHỤC OVERFITTING")
print("="*50)

model_fixed = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
model_fixed.fit(X_train, y_train)

y_train_pred_fixed = model_fixed.predict(X_train)
y_test_pred_fixed = model_fixed.predict(X_test)

r2_train_fixed = r2_score(y_train, y_train_pred_fixed)
r2_test_fixed = r2_score(y_test, y_test_pred_fixed)

print(f"R2 Score trên tập Train: {r2_train_fixed:.4f}")
print(f"R2 Score trên tập Test:  {r2_test_fixed:.4f}")
print(f"✅ Khoảng cách R2 (Gap): {r2_train_fixed - r2_test_fixed:.4f} (Nhỏ hơn nhiều, mô hình tổng quát hóa tốt)")

cv_scores = cross_val_score(model_fixed, X, y, cv=5, scoring='r2')
print(f"📊 R2 Score trung bình qua 5-fold Cross-Validation: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print("\n🎉 Hoàn thành! Mô hình đã được tối ưu để tránh Overfitting.")