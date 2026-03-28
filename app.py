import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Elderly Diabetes Risk Prediction")

# โหลดโมเดล
ensemble = joblib.load("models/ensemble.pkl")
mlp = joblib.load("models/nn.pkl")
scaler = joblib.load("models/scaler.pkl")

nn = joblib.load("models/nn_project2.pkl")
scaler2 = joblib.load("models/scaler_project2.pkl")

# MENU
page = st.sidebar.selectbox(
    "Menu",
    (
        "About Project",
        
        "Model 1) Explanation",
        "Diabetes Risk Prediction (Machine Learning)",
        "Model 1) Performance",
        
        "Model 2) Explanation",
        "Breast Cancer Prediction (Neural Network)",
        "Model 2) Performance",
        
        "References"
    )
)

# -----------------------------
# ABOUT PROJECT
# -----------------------------
if page == "About Project":

    st.title("🧠 AI Health Risk Prediction System")

    st.header("Project Overview")

    st.write("""
โครงการนี้เป็นการพัฒนา Web Application สำหรับวิเคราะห์และทำนายความเสี่ยง
ด้านสุขภาพ โดยใช้เทคนิค Machine Learning และ Neural Network

ภายในระบบมีการใช้ Dataset จำนวน 2 ชุด ได้แก่
- ข้อมูลโรคเบาหวาน (Diabetes Dataset)
- ข้อมูลมะเร็งเต้านม (Breast Cancer Dataset)

ข้อมูลทั้งสองชุดมีลักษณะไม่สมบูรณ์ในบางส่วน
จึงต้องผ่านกระบวนการ Data Preparation เช่น การทำความสะอาดข้อมูล
และการปรับสเกล (Feature Scaling) ก่อนนำไปใช้ในการพัฒนาโมเดล
""")

    st.header("Models Used")

    st.write("""
ระบบนี้พัฒนาโมเดลทั้งหมด 2 ประเภท ได้แก่

1. Ensemble Machine Learning Model  
   - ประกอบด้วยหลายโมเดล เช่น Decision Tree, Random Forest และ Gradient Boosting  
   - ใช้การรวมผลลัพธ์ (Ensemble) เพื่อเพิ่มความแม่นยำ  

2. Neural Network Model  
   - ออกแบบโครงสร้างโมเดลเอง  
   - สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนของข้อมูลได้ดี  
""")

    st.header("System Features")

    st.write("""
- วิเคราะห์ข้อมูลสุขภาพของผู้ใช้งาน  
- ทำนายความเสี่ยงของโรคเบาหวาน  
- ทำนายความเสี่ยงของมะเร็งเต้านม  
- เปรียบเทียบผลลัพธ์จาก Machine Learning และ Neural Network  
- ใช้งานผ่าน Web Application ได้ง่าย  
""")

    st.header("Objective")

    st.write("""
- ศึกษากระบวนการพัฒนา Machine Learning และ Neural Network  
- ฝึกการเตรียมข้อมูล (Data Preparation) จากข้อมูลที่ไม่สมบูรณ์  
- เปรียบเทียบประสิทธิภาพของโมเดลหลายประเภท  
- พัฒนา Web Application สำหรับใช้งานจริง  
""")

    st.warning("ระบบนี้เป็นเพียงเครื่องมือช่วยประเมินความเสี่ยงเบื้องต้น และไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")


# -----------------------------
# MODEL EXPLANATION
# -----------------------------
elif page == "Model 1) Explanation":
    

    st.title("📊 Model Explanation")

    st.header("Dataset")

    st.write("""

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
Ensemble Machine Learning เป็นเทคนิคที่นำโมเดลหลายตัวมาทำงานร่วมกัน
เพื่อเพิ่มประสิทธิภาพในการทำนาย โดยแนวคิดหลักคือ
การรวมจุดเด่นของแต่ละโมเดลเข้าด้วยกัน เพื่อลดข้อผิดพลาดของโมเดลเดี่ยว

แทนที่จะพึ่งพาการตัดสินใจจากโมเดลเพียงตัวเดียว
Ensemble จะใช้ผลลัพธ์จากหลายโมเดลมาประกอบกัน
เช่น การโหวต (Voting) หรือการเฉลี่ยผลลัพธ์

แนวทางนี้ช่วยให้โมเดลมีความแม่นยำสูงขึ้น
มีความเสถียรมากขึ้น และลดปัญหา Overfitting ได้อย่างมีประสิทธิภาพ
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
    
    st.image("images/decision_tree.png", caption="source: https://www.datacamp.com/tutorial/decision-tree-classification-python")

    st.write("""
Decision Tree เป็นโมเดล Machine Learning ที่มีโครงสร้างเป็นลักษณะต้นไม้
โดยแต่ละโหนด (Node) จะแทนเงื่อนไขในการตัดสินใจ
และแต่ละกิ่ง (Branch) จะแทนผลลัพธ์ของเงื่อนไขนั้น

โมเดลจะทำการแบ่งข้อมูลออกเป็นกลุ่มย่อย ๆ
ตามค่าของตัวแปร เช่น ระดับน้ำตาลในเลือด (Glucose)
หรือค่าดัชนีมวลกาย (BMI)

ข้อดีของ Decision Tree คือสามารถเข้าใจได้ง่าย
และสามารถอธิบายการตัดสินใจของโมเดลได้อย่างชัดเจน

อย่างไรก็ตาม โมเดลนี้มีข้อจำกัดคือ
อาจเกิด Overfitting ได้ง่าย หากต้นไม้มีความลึกมากเกินไป
""")
    st.subheader("Random Forest")
    
    st.image("images/random_forest.jpg",caption="source: https://datahacker.rs/012-machine-learning-introduction-to-random-forest/")

    st.write("""
Random Forest เป็นเทคนิค Ensemble ที่พัฒนามาจาก Decision Tree
โดยการสร้างต้นไม้หลายต้น (Multiple Decision Trees)

แต่ละต้นจะถูกฝึกด้วยชุดข้อมูลที่สุ่มขึ้นมา (Random Sampling)
และเลือกใช้คุณลักษณะ (Features) แบบสุ่มในแต่ละรอบ

เมื่อมีการทำนายผล โมเดลจะใช้วิธีการโหวต (Voting)
จากผลลัพธ์ของต้นไม้ทั้งหมด เพื่อหาคำตอบสุดท้าย

ข้อดีของ Random Forest คือ
สามารถลดปัญหา Overfitting ได้ดี
และให้ผลลัพธ์ที่มีความแม่นยำและเสถียรกว่า Decision Tree เพียงตัวเดียว

จึงเป็นโมเดลที่นิยมใช้ในงาน Machine Learning อย่างแพร่หลาย
""")
    
    st.subheader("Gradient Boosting")
    
    st.image("images/Gradient_Boosting.png",caption="source: https://datascience.eu/machine-learning/gradient-boosting-what-you-need-to-know/")

    st.write("""
Gradient Boosting เป็นเทคนิค Ensemble ที่สร้างโมเดลแบบลำดับขั้น (Sequential)
โดยแต่ละโมเดลใหม่จะถูกสร้างขึ้นเพื่อแก้ไขข้อผิดพลาดของโมเดลก่อนหน้า

โมเดลจะเรียนรู้จาก residual error หรือความผิดพลาด
แล้วปรับปรุงผลลัพธ์ให้ดีขึ้นในแต่ละรอบ

กระบวนการนี้ทำให้ Gradient Boosting
สามารถเรียนรู้ pattern ที่ซับซ้อนในข้อมูลได้ดี

ข้อดีคือมีความแม่นยำสูง
แต่ข้อเสียคือใช้เวลาในการฝึกโมเดลนาน
และต้องมีการปรับค่าพารามิเตอร์ (Hyperparameters) อย่างเหมาะสม
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
elif page == "Diabetes Risk Prediction (Machine Learning)":

    st.title("Diabetes Risk Prediction (Machine Learning)")

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
elif page == "Model 1) Performance":

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

elif page == "Model 2) Explanation":

    st.title("🎗️ Breast Cancer Prediction - Model Explanation")

    st.header("Dataset")

    st.write("""
โมเดลนี้ถูกพัฒนาจาก Breast Cancer Dataset
ซึ่งใช้ข้อมูลคุณลักษณะของเซลล์เนื้องอก
เพื่อทำนายว่าเป็นเนื้อร้าย (Malignant) หรือไม่ (Benign)
""")

    st.header("Features Used")

    st.write("""
ในโปรเจคนี้ได้เลือกใช้ feature ที่สำคัญจำนวน 5 ตัว ได้แก่

- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean

เหตุผลที่เลือก feature เหล่านี้เพราะเป็นตัวแทนของลักษณะทางกายภาพ
ของเซลล์ที่มีผลต่อการวินิจฉัยโรค
""")
    
    st.subheader("Neural Network")
     
    st.image("images/Neural_Networks.png", caption="source: https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/")

    st.write("""
Neural Network เป็นโมเดลที่ได้รับแรงบันดาลใจจากโครงสร้างของสมองมนุษย์
โดยประกอบด้วยโหนด (Nodes) หรือที่เรียกว่า Neurons
ซึ่งเชื่อมต่อกันเป็นชั้น (Layers)

โมเดลประกอบด้วย 3 ส่วนหลัก ได้แก่
Input Layer, Hidden Layers และ Output Layer

Neural Network สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนของข้อมูลได้
โดยใช้กระบวนการที่เรียกว่า Forward Propagation และ Backpropagation

ข้อดีของ Neural Network คือ
สามารถจัดการกับข้อมูลที่มีความซับซ้อนสูงได้ดี
เช่น ความสัมพันธ์ระหว่างหลายตัวแปรทางสุขภาพ

อย่างไรก็ตาม โมเดลนี้ต้องการข้อมูลจำนวนมาก
และใช้เวลาในการฝึกค่อนข้างสูง
เมื่อเทียบกับโมเดล Machine Learning แบบดั้งเดิม
""")

    st.header("Why Neural Network")

    st.write("""
Neural Network ถูกเลือกใช้เนื่องจากสามารถเรียนรู้ความสัมพันธ์
ที่ซับซ้อนของข้อมูลได้ดีกว่าโมเดลทั่วไป

ข้อดีของ Neural Network:

1. สามารถเรียนรู้ pattern ที่ซับซ้อนได้
2. รองรับข้อมูลหลายมิติได้ดี
3. เหมาะกับงานด้านการแพทย์ที่มีหลายปัจจัยร่วมกัน

ในโมเดลนี้ใช้โครงสร้าง:

- Input Layer (5 features)
- Hidden Layer 1 (64 neurons)
- Hidden Layer 2 (32 neurons)
- Output Layer (Binary Classification)
""")

    st.header("Data Preprocessing")

    st.write("""
ก่อนนำข้อมูลเข้าโมเดล ได้มีการทำ Feature Scaling
โดยใช้ StandardScaler

การ scaling ช่วยให้ค่าของแต่ละ feature อยู่ในช่วงที่ใกล้เคียงกัน
ทำให้ Neural Network เรียนรู้ได้มีประสิทธิภาพมากขึ้น
""")

    st.header("Prediction Workflow")

    st.write("""
ขั้นตอนการทำงานของระบบ:

1. รับ input จากผู้ใช้
2. ทำการ scaling ข้อมูล
3. ส่งข้อมูลเข้า Neural Network
4. ทำนายผลว่าเป็น Malignant หรือ Benign
""")
    
elif page == "Breast Cancer Prediction (Neural Network)":

    st.title("🎗️ Breast Cancer Prediction (Neural Network)")

    model = joblib.load("models/nn_project2.pkl")
    scaler = joblib.load("models/scaler_project2.pkl")

    radius = st.number_input("Radius Mean", 5.0, 30.0, 14.0)
    texture = st.number_input("Texture Mean", 5.0, 40.0, 20.0)
    perimeter = st.number_input("Perimeter Mean", 40.0, 200.0, 90.0)
    area = st.number_input("Area Mean", 100.0, 2500.0, 500.0)
    smoothness = st.number_input("Smoothness Mean", 0.05, 0.2, 0.1)

    if st.button("Predict Cancer Risk"):

        input_data = np.array([[radius, texture, perimeter, area, smoothness]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]

        if pred == 1:
            st.error("⚠️ High Risk (Malignant)")
        else:
            st.success("✅ Low Risk (Benign)")
            
elif page == "Model 2) Performance":

    st.title("📈 Model 2 Performance")

    st.write("""
โมเดล Neural Network ถูกประเมินผลด้วยชุดข้อมูลทดสอบ
(Test Set) เพื่อวัดความสามารถในการทำนาย
""")

    st.subheader("Accuracy")

    st.write("""
Accuracy: 0.9649
""")

    st.write("""
โมเดลสามารถทำนายได้อย่างแม่นยำสูงถึง 96.49%

แสดงให้เห็นว่า Neural Network สามารถเรียนรู้ pattern
ของข้อมูลทางการแพทย์ได้อย่างมีประสิทธิภาพ

ค่าความแม่นยำที่สูงนี้บ่งบอกว่าโมเดลมีความสามารถ
ในการแยกแยะระหว่างเนื้อร้าย (Malignant)
และเนื้อไม่ร้ายแรง (Benign) ได้ดี
""")

# -----------------------------
# REFERENCES
# -----------------------------
elif page == "References":

    st.title("📚 References")

    st.write("""
1. Kaggle Dataset  
https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


2. Scikit-learn Documentation  
https://scikit-learn.org
""")