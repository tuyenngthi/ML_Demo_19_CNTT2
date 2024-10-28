import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib #thư viện để lưu mô hình
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Hàm tính NSE
def tinh_nse(y_test, y_pred):
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# Hàm tính MAE thủ công
def error(y, y_pred):
    return np.mean(np.abs(y - y_pred))

# Đọc dữ liệu từ file
data = pd.read_csv('data.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80-20)
dt_Train, dt_Test = train_test_split(data, test_size=0.2, shuffle=False)

# Xử lý các giá trị thiếu trong tập kiểm tra nếu có
imputer = SimpleImputer(strategy='mean')
if dt_Test.isnull().values.any():
    dt_Test_imputed = imputer.fit_transform(dt_Test)
else:
    dt_Test_imputed = dt_Test.values  # Chuyển đổi dt_Test thành mảng NumPy nếu không có giá trị thiếu

# Khởi tạo giá trị lỗi tối thiểu để lưu mô hình tốt nhất
min_error = float("inf")
k = 10  # Số lượng fold
kf = KFold(n_splits=k, random_state=None, shuffle=True)

# K-fold cross-validation
for train_index, validation_index in kf.split(dt_Train):
    X_train, X_validation = dt_Train.iloc[train_index, 1:], dt_Train.iloc[validation_index, 1:]
    y_train, y_validation = dt_Train.iloc[train_index, 0], dt_Train.iloc[validation_index, 0]

    # Xử lý các giá trị thiếu trong tập train và validation
    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)

    # Huấn luyện mô hình
    lr = LinearRegression().fit(X_train_imputed, y_train)
    y_train_pred = lr.predict(X_train_imputed)
    y_validation_pred = lr.predict(X_validation_imputed)

    # Tính tổng lỗi MAE trên tập train và validation
    sum_error = error(y_train, y_train_pred) + error(y_validation, y_validation_pred)
    
    # Lưu lại mô hình nếu có lỗi nhỏ hơn
    if sum_error < min_error:
        min_error = sum_error
        best_model = lr

# Dự đoán trên tập kiểm tra
y_test_pred = best_model.predict(dt_Test_imputed[:, 1:])
y_test = np.array(dt_Test.iloc[:, 0])

# In các chỉ số đánh giá mô hình
print("R2: %.9f" % r2_score(y_test, y_test_pred))
print("NSE: %.9f" % tinh_nse(y_test, y_test_pred))
print("MAE: %.9f" % mean_absolute_error(y_test, y_test_pred))
print("RMSE: %.9f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))

# In ra sự chênh lệch giữa giá trị thực tế và giá trị dự đoán
print("Thuc te  | Du doan  | Chenh lech")
for i in range(len(y_test)):
    print(f"{y_test[i]:<10.3f} {y_test_pred[i]:<10.3f} {abs(y_test[i] - y_test_pred[i]):<10.3f}")

# Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán
plt.figure(figsize=(12, 6))
x = range(len(y_test))  # Chỉ số mẫu

# Biểu đồ đường thể hiện giá trị thực tế và giá trị dự đoán
plt.plot(x, y_test, marker='o', color='b', label='Giá trị thực tế')
plt.plot(x, y_test_pred, marker='x', color='r', linestyle='--', label='Giá trị dự đoán')

# Thiết lập tiêu đề và nhãn cho biểu đồ
plt.xlabel('Mẫu')
plt.ylabel('Giá trị')
plt.title('So sánh giữa Giá trị thực tế và Giá trị dự đoán')
plt.legend()
plt.grid(True)
plt.show()

#lưu mô hình vào file
joblib.dump(best_model, 'k_fold_model.pkl')