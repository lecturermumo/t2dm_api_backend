from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)
booster = xgb.Booster()
booster.load_model("xgboost_model_booster.json")

feature_names = [
    "Age", "BMI", "Systolic_BP", "Diastolic_BP", "Fasting_Glucose",
    "Family_History", "Physical_Activity", "Alcohol_Use",
    "Education_Level", "Income_Level", "Smoking"
]

def categorize_risk(prob):
    if prob <= 0.33:
        return "Low"
    elif prob <= 0.66:
        return "Medium"
    else:
        return "High"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [data.get(f) for f in feature_names]
        df = pd.DataFrame([features], columns=feature_names)
        dmatrix = xgb.DMatrix(df, feature_names=feature_names)
        prob = booster.predict(dmatrix)[0]
        risk = categorize_risk(prob)
        return jsonify({"risk": risk, "score": round(prob * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # <-- allow all origins by default
CORS(app, origins=["https://isaacmumo.co.ke"])
