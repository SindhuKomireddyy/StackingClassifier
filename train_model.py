import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/heart.csv")

# =====================================================
# Feature Engineering
# =====================================================

df["Age_Cholesterol"] = df["Age"] * df["Cholesterol"]

df["BP_Age"] = df["RestingBP"] * df["Age"]

df["MaxHR_Age_Ratio"] = df["MaxHR"] / (df["Age"] + 1)

# =====================================================
# Outlier Treatment
# =====================================================

cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
]

for col in cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# =====================================================
# Encoding
# =====================================================

categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

pickle.dump(
    encoders,
    open("models/encoders.pkl", "wb")
)

# =====================================================
# Features and Target
# =====================================================

X = df.drop("HeartDisease", axis=1)

y = df["HeartDisease"]

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

# =====================================================
# Base Learners
# =====================================================

base_models = [

    (
        "lr",
        LogisticRegression(
            C=1,
            max_iter=500
        )
    ),

    (
        "dt",
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=8,
            random_state=42
        )
    )

]

# =====================================================
# Meta Learner
# =====================================================

meta_model = LogisticRegression(
    C=0.5,
    max_iter=500
)

# =====================================================
# Stacking Classifier
# =====================================================

model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

model.fit(X_train, y_train)

# =====================================================
# Evaluation
# =====================================================

y_pred = model.predict(X_test)

print("Accuracy :", round(
    accuracy_score(y_test, y_pred), 4
))

print(classification_report(
    y_test,
    y_pred
))

# =====================================================
# Save Model
# =====================================================

pickle.dump(
    model,
    open(
        "models/stacking_classifier.pkl",
        "wb"
    )
)

print("Model Saved Successfully")