import streamlit as st
import numpy as np
import joblib

# load models
ensemble = joblib.load("models/ensemble.pkl")
mlp = joblib.load("models/nn.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Elderly Diabetes Risk Prediction")

st.title("🧓 Elderly Diabetes Risk Prediction")
st.write("ระบบทำนายความเสี่ยงโรคเบาหวานในผู้สูงอายุ")

# sidebar
model_choice = st.sidebar.selectbox(
    "เลือกโมเดลที่ใช้ทำนาย",
    ("Ensemble Machine Learning", "Neural Network")
)

st.header("กรอกข้อมูลสุขภาพ")

preg = st.number_input("Pregnancies", 0, 20, 0)
glu = st.number_input("Glucose", 50, 300, 120)
bp = st.number_input("Blood Pressure", 30, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 50, 100, 60)

if st.button("🔍 Predict Risk"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "Ensemble Machine Learning":
        prediction = ensemble.predict(input_scaled)[0]
    else:
        prediction = mlp.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ มีความเสี่ยงเป็นโรคเบาหวาน")
    else:
        st.success("✅ ความเสี่ยงต่ำ")

    st.info("ผลลัพธ์นี้เป็นการประเมินเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์")