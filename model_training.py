import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle  # לשמירת המודל
import os

try:
    from assets_data_prep import prepare_data
except Exception as e:
    print("Import failed:", e)
    raise

try:
    from sklearn.metrics import root_mean_squared_error
    SKLEARN_NEW = True
except ImportError:
    SKLEARN_NEW = False

# --- מזהה הקבוצה ---
id = "Team 7"

# --- 1. קריאה והכנת דאטה ---
print("📊 טוען ומעבד נתונים...")
df = pd.read_csv("train.csv")
print(f"📊 גודל הדאטה המקורי: {df.shape}")

# הכנה בסיסית בלבד (ללא Feature Engineering מתקדם)
df_prepared = prepare_data(df, "train")  # 
print(f"📊 גודל הדאטה לאחר prepare_data: {df_prepared.shape}")

X = df_prepared.drop(columns=["price"])
y = df_prepared["price"]

# ניקוי ערכים חסרים
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# --- 2. חלוקה לסטים ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. ElasticNet ---
print("🔧 מאמן מודל ElasticNet...")
elastic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(cv=5, random_state=42))
])
elastic_pipeline.fit(X_train, y_train)
y_pred_elastic = elastic_pipeline.predict(X_test)

# שמירה למודל ולפיצ'רים
with open("trained_model.pkl", "wb") as f:
    pickle.dump(elastic_pipeline, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("💾 ElasticNet נשמר.")



