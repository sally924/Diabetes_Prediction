import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Elderly Diabetes Risk Prediction")

# Sidebar menu
page = st.sidebar.selectbox(
    "Menu",
    ("Prediction", "Model Explanation")
)

# load models
ensemble = joblib.load("models/ensemble.pkl")
mlp = joblib.load("models/nn.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# PAGE 1 : Prediction
# -----------------------------
if page == "Prediction":

    st.title("🧓 Elderly Diabetes Risk Prediction")
    st.write("ระบบทำนายความเสี่ยงโรคเบาหวานในผู้สูงอายุ")

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


# -----------------------------
# PAGE 2 : Model Explanation
# -----------------------------
elif page == "Model Explanation":

    st.title("📊 Model Explanation")

    st.header("1. Purpose of the Model")
    st.write("""
โมเดลนี้ถูกพัฒนาขึ้นเพื่อช่วยประเมินความเสี่ยงของโรคเบาหวานในผู้สูงอายุ  
ระบบจะรับข้อมูลสุขภาพของผู้ใช้ แล้วนำไปผ่านโมเดล Machine Learning  
เพื่อคำนวณความน่าจะเป็นของการเป็นโรคเบาหวาน
""")

    st.header("2. Dataset")

    st.write("""
โมเดลนี้ฝึกด้วย **Pima Indians Diabetes Dataset**

Dataset นี้ประกอบด้วยข้อมูลสุขภาพ เช่น

- ระดับน้ำตาลในเลือด
- ความดันโลหิต
- BMI
- อายุ
- ประวัติครอบครัวเกี่ยวกับโรคเบาหวาน
""")

    st.header("3. Input Features")

    st.markdown("""
โมเดลใช้ตัวแปรทั้งหมด **8 ตัว**

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

    st.header("4. Data Preprocessing")

    st.write("""
ก่อนส่งข้อมูลเข้าโมเดล ระบบจะทำ **Feature Scaling**

เราใช้ **StandardScaler**

เพื่อทำให้ข้อมูลทุกตัวแปรมีช่วงค่าใกล้เคียงกัน  
ซึ่งช่วยให้โมเดลเรียนรู้ได้มีประสิทธิภาพมากขึ้น
""")

    st.header("5. Machine Learning Models")

    st.subheader("Ensemble Machine Learning")

    st.write("""
Ensemble Model คือการรวมหลายโมเดลเข้าด้วยกัน

ข้อดี

- เพิ่มความแม่นยำ
- ลด Overfitting
- ทำให้การทำนายมีความเสถียรมากขึ้น
""")

    st.subheader("Neural Network")

    st.write("""
Neural Network เป็นโมเดลที่เลียนแบบโครงสร้างสมองมนุษย์

ประกอบด้วย

- Input Layer
- Hidden Layer
- Output Layer

โมเดลสามารถเรียนรู้ pattern ที่ซับซ้อนในข้อมูลได้
""")

    st.header("6. Prediction Workflow")

    st.markdown("""
ขั้นตอนการทำงานของระบบ

1. ผู้ใช้กรอกข้อมูลสุขภาพ  
2. ระบบทำ Data Scaling  
3. ส่งข้อมูลเข้าสู่ Machine Learning Model  
4. โมเดลทำนายความเสี่ยงโรคเบาหวาน
""")

    st.header("7. Limitations")

    st.warning("""
ระบบนี้เป็นเพียงเครื่องมือช่วยประเมินความเสี่ยงเบื้องต้น  
ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้
""")