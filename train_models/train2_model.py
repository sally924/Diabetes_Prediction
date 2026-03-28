import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# โหลดข้อมูล
df = pd.read_csv("data.csv")

# ลบ column ที่ไม่ใช้
df = df.drop(columns=["id", "Unnamed: 32"])

# แปลง label
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


selected_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean"
]

X = df[selected_features]
y = df["diagnosis"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500
)

# train
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# save
joblib.dump(model, "nn_project2.pkl")
joblib.dump(scaler, "scaler_project2.pkl")