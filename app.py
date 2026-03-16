import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Elderly Diabetes Risk Prediction")

# โหลดโมเดล
ensemble = joblib.load("models/ensemble.pkl")
mlp = joblib.load("models/nn.pkl")
scaler = joblib.load("models/scaler.pkl")

# MENU
page = st.sidebar.selectbox(
    "Menu",
    (
        "About Project",
        "Model Explanation",
        "Prediction",
        "Model Performance",
        "References"
    )
)

# -----------------------------
# ABOUT PROJECT
# -----------------------------
if page == "About Project":

    st.title("🧓 Elderly Diabetes Risk Prediction")

    st.header("Project Overview")

    st.write("""
โครงการนี้พัฒนาระบบ Machine Learning เพื่อช่วยประเมินความเสี่ยง
ของโรคเบาหวานในผู้สูงอายุ

ผู้ใช้สามารถกรอกข้อมูลสุขภาพ เช่น ระดับน้ำตาลในเลือด
ค่าดัชนีมวลกาย และอายุ จากนั้นระบบจะใช้โมเดล AI
ในการประเมินความเสี่ยงเบื้องต้น
""")

    st.header("Objective")

    st.write("""
วัตถุประสงค์ของระบบคือ

- ช่วยประเมินความเสี่ยงของโรคเบาหวาน
- ใช้ Machine Learning วิเคราะห์ข้อมูลสุขภาพ
- เป็นเครื่องมือช่วยคัดกรองเบื้องต้น
""")

    st.warning("ระบบนี้ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")


# -----------------------------
# MODEL EXPLANATION
# -----------------------------
elif page == "Model Explanation":
    

    st.title("📊 Model Explanation")

    st.header("Dataset")

    st.write("""
โมเดลนี้ถูกฝึกด้วย **Pima Indians Diabetes Dataset**

Dataset นี้ประกอบด้วยข้อมูลสุขภาพของผู้ป่วย เช่น

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

    st.header("Features Used")

    st.write("""
โมเดลใช้ตัวแปรทั้งหมด 8 ตัวในการทำนาย

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age
""")

    st.header("Data Preprocessing")

    st.write("""
ก่อนส่งข้อมูลเข้าโมเดล ระบบจะทำ **Feature Scaling**
โดยใช้ StandardScaler

การทำ scaling จะช่วยให้ค่าของตัวแปรอยู่ในช่วงใกล้เคียงกัน
ซึ่งช่วยให้โมเดลเรียนรู้ได้มีประสิทธิภาพมากขึ้น
""")

    st.header("Machine Learning Models")

    st.subheader("Ensemble Machine Learning")

    st.write("""
Ensemble เป็นเทคนิคที่รวมหลายโมเดลเข้าด้วยกัน
เพื่อเพิ่มความแม่นยำในการทำนาย
""")
    
    st.subheader("Ensemble Model Structure")

    st.write("""
โมเดล Ensemble ในระบบนี้ประกอบด้วย Machine Learning หลายโมเดล ได้แก่

• Decision Tree  
• Random Forest  
• Gradient Boosting

แต่ละโมเดลจะทำการเรียนรู้ข้อมูลแยกกัน
จากนั้นระบบจะรวมผลลัพธ์ของทุกโมเดลเข้าด้วยกัน
เพื่อให้ได้ผลลัพธ์ที่แม่นยำมากขึ้น
""")
    st.subheader("Decision Tree")

    st.write("""
Decision Tree เป็นโมเดลที่ทำการตัดสินใจในรูปแบบโครงสร้างต้นไม้
โดยจะแบ่งข้อมูลตามเงื่อนไขของตัวแปรต่าง ๆ เช่น
ระดับน้ำตาลในเลือด หรือค่า BMI
""")
    st.subheader("Random Forest")

    st.write("""
Random Forest เป็นการรวม Decision Tree หลายต้นเข้าด้วยกัน
โดยแต่ละต้นจะเรียนรู้จากข้อมูลที่แตกต่างกันเล็กน้อย

ผลลัพธ์สุดท้ายจะใช้วิธีการโหวต (Voting)
ซึ่งช่วยลดปัญหา Overfitting และเพิ่มความเสถียรของโมเดล
""")
    
    st.subheader("Gradient Boosting")

    st.write("""
Gradient Boosting เป็นโมเดลที่สร้าง Decision Tree หลายต้น
โดยแต่ละต้นจะพยายามแก้ไขข้อผิดพลาดของต้นก่อนหน้า

วิธีนี้ช่วยให้โมเดลสามารถเรียนรู้ pattern ของข้อมูลได้ดีขึ้น
""")

    st.subheader("Neural Network")

    st.write("""
Neural Network เป็นโมเดลที่เลียนแบบโครงสร้างของสมองมนุษย์
และสามารถเรียนรู้ pattern ที่ซับซ้อนในข้อมูลได้
""")

    st.header("Prediction Workflow")

    st.write("""
ขั้นตอนการทำงานของระบบ

1. ผู้ใช้กรอกข้อมูลสุขภาพ
2. ระบบทำ Data Scaling
3. ส่งข้อมูลเข้าสู่ Machine Learning Model
4. โมเดลทำนายความเสี่ยงโรคเบาหวาน
""")


# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":

    st.title("🔍 Diabetes Risk Prediction")

    model_choice = st.selectbox(
        "เลือกโมเดลที่ใช้ทำนาย",
        ("Ensemble Machine Learning", "Neural Network")
    )

    preg = st.number_input("Pregnancies", 0, 20, 0)
    glu = st.number_input("Glucose", 50, 300, 120)
    bp = st.number_input("Blood Pressure", 30, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    ins = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 50, 100, 60)

    if st.button("Predict Risk"):

        input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])

        input_scaled = scaler.transform(input_data)

        if model_choice == "Ensemble Machine Learning":
            prediction = ensemble.predict(input_scaled)[0]
        else:
            prediction = mlp.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk")

        st.info("This result is only for preliminary screening.")


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write("""
โมเดลถูกประเมินผลด้วยชุดข้อมูลทดสอบ
เพื่อวัดประสิทธิภาพของการทำนาย
""")

    st.write("""
Model Accuracy Results

Decision Tree: 0.67  
Random Forest: 0.85  
Gradient Boosting: 0.67  
Ensemble Model: 0.74  
Neural Network: 0.74
""")

# -----------------------------
# REFERENCES
# -----------------------------
elif page == "References":

    st.title("📚 References")

    st.write("""
1. Kaggle Dataset  
Pima Indians Diabetes Dataset  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

2. Scikit-learn Documentation  
https://scikit-learn.org
""")