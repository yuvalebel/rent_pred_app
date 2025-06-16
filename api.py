from flask import Flask, render_template, request
import pandas as pd
import joblib
from assets_data_prep import prepare_data
import os

app = Flask(__name__)

if not (os.path.exists('trained_model.pkl') and os.path.exists('feature_columns.pkl')):
    from model_training import *

import pickle

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {k: [v] for k, v in request.form.items()}
    df = pd.DataFrame(data)
    X = prepare_data(df, dataset_type="test")
    # ודא שכל הפיצ'רים קיימים ובסדר הנכון, ומחק עמודות עודפות
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]  # שמור רק את הפיצ'רים שהמודל מצפה להם ובסדר הנכון
    # מחק עמודות עודפות אם יש
    X = X.loc[:, feature_names]
    prediction = model.predict(X)[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
