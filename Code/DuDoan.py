import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Tải các mô hình đã lưu
models = {
    'k-Fold Cross-Validation': joblib.load('k_fold_model.pkl'),
    'Ridge Regression': joblib.load('ridge_model.pkl'),
    'Linear Regression': joblib.load('linear_regression_model.pkl'),
    'Lasso Model': joblib.load('lasso_model.pkl')
}

# 2. Đọc dữ liệu gốc để lấy giá trị trung vị cho các cột số
data = pd.read_csv('data.csv')

# 3. Lấy danh sách các đặc trưng
numeric_features = [
    'Bedroom', 'Space', 'Room', 'Lot', 'Tax',
    'Bathroom', 'Garage', 'Condition'
]

# 4. Tạo giao diện người dùng
st.title('🏠 Dự đoán giá nhà')

st.write('Nhập thông tin bên dưới để dự đoán giá nhà:')

# 5. Tạo các input cho người dùng
input_data = {}

# Nhập các đặc trưng số
for feature in numeric_features:
    input_value = st.text_input(f'{feature}', '')
    input_data[feature] = input_value

# Lựa chọn mô hình
model_name = st.selectbox(
    'Chọn mô hình để dự đoán',
    options=list(models.keys()),
    index=3  # Mặc định chọn Lasso Model
)

# 6. Khi người dùng nhấn nút 'Dự đoán'
if st.button('Dự đoán'):
    # Chuyển đổi input_data thành DataFrame
    input_df = pd.DataFrame([input_data])

    # Xử lý các ô không nhập dữ liệu
    # Chuyển các giá trị số từ chuỗi sang số thực
    for col in numeric_features:
        if input_df[col][0] == '':
            st.warning(f'Bạn chưa nhập giá trị cho {col}. Sẽ sử dụng giá trị trung vị.')
            input_df[col] = data[col].median()
        else:
            try:
                input_df[col] = float(input_df[col][0])
            except ValueError:
                st.error(f'Giá trị nhập vào cho {col} không hợp lệ. Vui lòng nhập số.')
                st.stop()

    # Lấy mô hình được chọn
    selected_model = models[model_name]

    # Dự đoán
    prediction = selected_model.predict(input_df)[0]

    # Hiển thị kết quả
    st.success(f'Mô hình sử dụng: **{model_name}**')
    st.success(f'Giá nhà dự đoán: **${prediction:,.2f}**')
