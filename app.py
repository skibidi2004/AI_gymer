
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load preprocessor & model
with open('./models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('./models/linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Cấu hình giao diện
st.set_page_config(page_title="🔋 Dự đoán Calories tiêu hao", layout="wide")

# CSS tùy chỉnh
st.markdown("""
    <style>
        .st-emotion-cache-br351g p, .st-emotion-cache-br351g ol, .st-emotion-cache-br351g ul, .st-emotion-cache-br351g dl, .st-emotion-cache-br351g li {
            font-size: 24px !important;
        }
        label.css-1cpxqw2, label.css-1pyh2j1 {
            font-size: 28px !important;
            font-weight: 700 !important;
        }
        .stButton > button {
            font-size: 28px !important;
            padding: 0.7em 2.5em !important;
            border-radius: 10px;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton > button:hover {
            background-color: #e63c3c;
        }
        .css-1hyem7o {
            font-size: 24px !important;
        }
        .stForm > div {
            margin-bottom: 0.75em;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            height: 60px !important;
            font-size: 28px !important;
        }
        .stForm {
            max-width: 1000px;
            margin: auto;
        }
        .stColumn {
            padding-right: 15px;
            padding-left: 15px;
        }
        .stTextInput label, .stNumberInput label, .stSelectbox label {
            font-size: 28px !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề
st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #4F8BF9; font-size: 40px;'>🔋 DỰ ĐOÁN LƯỢNG CALORIES TIÊU HAO</h1>
        <p style='font-size: 22px;'>Ứng dụng giúp bạn ước tính lượng calories tiêu hao trong mỗi buổi tập</p>
    </div>
""", unsafe_allow_html=True)

# Giao diện nhập liệu
with st.form("input_form"):
    st.subheader("📋 Nhập thông tin của bạn:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Tuổi", min_value=3, max_value=100, value=25)
        gender = st.selectbox("🚻 Giới tính", ["Male", "Female"])
        weight = st.number_input("⚖️ Cân nặng (kg)", min_value=10.0, max_value=200.0, value=60.0)
        height = st.number_input("📏 Chiều cao (m)", min_value=1.2, max_value=2.2, value=1.65)
        max_bpm = st.number_input("❤️ Max BPM", min_value=100, max_value=250, value=180)
        avg_bpm = st.number_input("💓 Avg BPM", min_value=60, max_value=200, value=120)

    with col2:
        resting_bpm = st.number_input("🛌 Resting BPM", min_value=40, max_value=120, value=70)
        session_duration = st.number_input("⏱️ Thời lượng tập (giờ)", min_value=0.5, max_value=4.0, value=1.0)
        workout_type = st.selectbox("🏋️ Loại bài tập", ["Yoga", "HIIT", "Cardio", "Strength"])
        fat_percentage = st.number_input("⚠️ Tỉ lệ mỡ (%)", min_value=5.0, max_value=50.0, value=20.0)
        water_intake = st.number_input("💧 Lượng nước (lít)", min_value=0.5, max_value=5.0, value=2.0)
        workout_freq = st.number_input("📅 Số buổi/tuần", min_value=1, max_value=7, value=3)
        experience_level = st.selectbox("🎖️ Cấp độ kinh nghiệm", [1, 2, 3], format_func=lambda x: f"Cấp {x}")

    # BMI
    bmi = round(weight / (height ** 2), 2)
    st.markdown(f"<p style='background-color:#e8f4fd;padding:10px;border-radius:10px'><b>💡 BMI của bạn:</b> {bmi}</p>", unsafe_allow_html=True)

    # Submit
    submitted = st.form_submit_button("🚀 Dự đoán Calories")

# Xử lý khi nhấn nút
if submitted:
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

    # Tiền xử lý và dự đoán
    X = preprocessor.transform(input_df)
    calories = model.predict(X)[0]
    st.markdown(f"""
    <div style='text-align: center; background-color: #fff4e6; padding: 30px; border-radius: 15px; margin-top: 20px;'>
        <h2 style='color: #e67300; font-size: 44px;'>🔥 Lượng calories tiêu hao dự đoán:</h2>
        <h1 style='font-size: 56px; color: #ff6600;'>{calories:.2f} kcal</h1>
    </div>
    """, unsafe_allow_html=True)

