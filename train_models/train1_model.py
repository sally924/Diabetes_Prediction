import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

#https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download
df = pd.read_csv('diabetes.csv')

elderly_df = df[df['Age'] >= 50]

cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
elderly_df[cols_with_zero] = elderly_df[cols_with_zero].replace(0,np.nan)
elderly_df.isnull().sum()


for col in cols_with_zero:
    elderly_df[col].fillna(elderly_df[col].median(), inplace=True)

X = elderly_df.drop('Outcome', axis=1)
y = elderly_df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#elderly_df.shape
#elderly_df['Outcome'].value_counts()

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

dt_acc = dt.score(X_test,y_test)
print("Decision Tree accuracy: ", dt_acc)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)

rf_acc = rf.score(X_test,y_test)
print("Random Forest accuracy: ", rf_acc)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train,y_train)

gb_acc = gb.score(X_test,y_test)
print("Gradient Boosting accuracy: ", gb_acc)

ensemble = VotingClassifier(
    estimators=[
        ('dt',dt),
        ('rf',rf),
        ('gb',gb)
    ],
    voting='hard'
)

ensemble.fit(X_train,y_train)

ensemble_acc = ensemble.score(X_test,y_test)
print("Ensemble accuracy: ", ensemble_acc)

mlp = MLPClassifier(
    hidden_layer_sizes=(64,32),
    max_iter=500,
    random_state=42
)

mlp.fit(X_train,y_train)

mlp_acc = mlp.score(X_test,y_test)
print("Neural Network accuracy: ", mlp_acc)

joblib.dump(ensemble,'ensemble_model.pkl')
joblib.dump(mlp,'nn_model.pkl')
joblib.dump(scaler,'scaler.pkl')
