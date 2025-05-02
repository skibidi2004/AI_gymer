
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load preprocessor & model
with open('./models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('./models/linear.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Dự đoán Calories tiêu hao", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>Dự đoán Calories tiêu hao</h1>", unsafe_allow_html=True)

# Giao diện nhập dữ liệu
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Tuổi", min_value=3, max_value=100, value=25)
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        weight = st.number_input("Cân nặng (kg)", min_value=10.0, max_value=200.0, value=60.0)
        height = st.number_input("Chiều cao (m)", min_value=1.2, max_value=2.2, value=1.65)
    with col2:
        max_bpm = st.number_input("Max BPM", min_value=100, max_value=250, value=180)
        avg_bpm = st.number_input("Avg BPM", min_value=60, max_value=200, value=120)
        resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=120, value=70)
        session_duration = st.number_input("Thời lượng buổi tập (giờ)", min_value=0.5, max_value=4.0, value=1.0)
    with col3:
        workout_type = st.selectbox("Loại bài tập", ["Yoga", "HIIT", "Cardio", "Strength"])
        fat_percentage = st.number_input("Tỉ lệ mỡ (%)", min_value=5.0, max_value=50.0, value=20.0)
        water_intake = st.number_input("Lượng nước (lít)", min_value=0.5, max_value=5.0, value=2.0)
        workout_freq = st.number_input("Số buổi/tuần", min_value=1, max_value=7, value=3)
        experience_level = st.selectbox("Cấp độ kinh nghiệm", [1, 2, 3])

    # Tính BMI tự động
    bmi = round(weight / (height ** 2), 2)
    st.info(f"**Chỉ số BMI:** {bmi}")

    submitted = st.form_submit_button("Dự đoán Calories")

if submitted:
    # Chuẩn bị dữ liệu
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration (hours)": session_duration,
        "Workout_Type": workout_type,
        "Fat_Percentage": fat_percentage,
        "Water_Intake (liters)": water_intake,
        "Workout_Frequency (days/week)": workout_freq,
        "Experience_Level": experience_level,
    }])

    # Tiền xử lý & dự báo
    X = preprocessor.transform(input_df)
    calories = model.predict(X)[0]
    st.success(f"**Lượng calories tiêu hao dự đoán:** {calories:.2f} kcal")


