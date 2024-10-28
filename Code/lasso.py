import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import joblib #thư viện để lưu mô hình

# Đọc dữ liệu từ file
data = pd.read_csv('data.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra 2-8
dt_Train, dt_Test = train_test_split(data, test_size=0.2, shuffle=False)

# Chọn đặc trưng X và biến mục tiêu y
X_train = dt_Train.iloc[:, 1:].values
y_train = dt_Train.iloc[:, 0].values
X_test = dt_Test.iloc[:, 1:].values
y_test = dt_Test.iloc[:, 0].values

# Tìm và xử lý các giá trị thiếu
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

clf = linear_model.Lasso(alpha=1).fit(X_train_imputed, y_train)
y_pred = clf.predict(X_test_imputed)
y = np.array(y_test)

print("R2: %.9f" % r2_score(y_test, y_pred))

print("Thuc te Du doan Chenh lech")
for i in range(0, len(y)):
    print("%.f" % y[i], "  %.2f" %y_pred[i], "  %.2f" %abs(y[i]-y_pred[i]))

# Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(12, 6))
x = range(len(y_test))  # Chỉ số mẫu

# Biểu đồ đường thể hiện giá trị thực tế và giá trị dự đoán
plt.plot(x, y_test, marker='o', color='b', label='Giá trị thực tế')
plt.plot(x, y_pred, marker='x', color='r', linestyle='--', label='Giá trị dự đoán')

# Thiết lập tiêu đề và nhãn cho biểu đồ
plt.xlabel('Mẫu')
plt.ylabel('Giá trị')
plt.title('So sánh giữa Giá trị Thực tế và Giá trị Dự đoán')
plt.legend()
plt.grid(True)
plt.show()

#lưu mô hình vào file
joblib.dump(clf, 'lasso_model.pkl')