import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
data = pd.read_excel("Housing.xlsx")

# X: diện tích
X = data[["area"]]

# y: giá nhà
y = data["price"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự báo
y_pred = model.predict(X_test)

# In kết quả
print("Hệ số góc:", model.coef_[0])
print("Hệ số chặn:", model.intercept_)

# Nhập diện tích cần dự báo
dien_tich = float(input("Nhập diện tích nhà: "))

gia_du_bao = model.predict([[dien_tich]])

print("Giá nhà dự báo:", gia_du_bao[0])