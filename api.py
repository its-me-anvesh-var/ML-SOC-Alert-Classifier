#!/usr/bin/env python3
"""
============================================================
api.py — SOC Alert Classifier REST API
Author: Anvesh Raju Vishwaraju
Usage: python src/api.py
       curl -X POST http://localhost:5000/predict -H "Content-Type: application/json"
            -d '{"severity":3,"source_ip_reputation":85,...}'
============================================================
"""
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Load model at startup
MODEL_PATH = "models/alert_classifier.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    clf = model_data["model"]
    FEATURES = model_data["features"]
    print(f"[✓] Model loaded. Features: {FEATURES}")
except FileNotFoundError:
    print("[!] Model not found. Run: python src/train.py first.")
    clf = None
    FEATURES = []


def engineer_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as training."""
    df = pd.DataFrame([data])
    df["bytes_log"] = np.log1p(df.get("bytes_transferred", 0))
    df["is_after_hours"] = ((df["hour_of_day"] < 7) |
                             (df["hour_of_day"] > 21)).astype(int)
    df["risk_score"] = (
        df["severity"] * 0.3 +
        df["source_ip_reputation"] / 100 * 0.3 +
        df["failed_logins_last_hour"] / 50 * 0.2 +
        df["is_admin_account"] * 0.2
    )
    return df[FEATURES]


def format_prediction(pred: int, prob: float) -> dict:
    """Format prediction with human-readable output."""
    label = "TRUE POSITIVE" if pred == 1 else "FALSE POSITIVE"
    confidence = round(prob * 100, 2)

    if pred == 1:
        if confidence >= 90:
            priority = "P1 - CRITICAL"
            action = "Escalate immediately to senior analyst"
        elif confidence >= 75:
            priority = "P2 - HIGH"
            action = "Investigate within 30 minutes"
        else:
            priority = "P3 - MEDIUM"
            action = "Review within 2 hours"
    else:
        priority = "INFO"
        action = "Auto-close or queue for batch review"

    return {
        "prediction": label,
        "confidence": f"{confidence}%",
        "priority": priority,
        "recommended_action": action
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": clf is not None,
        "timestamp": datetime.now().isoformat(),
        "author": "Anvesh Raju Vishwaraju"
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "Random Forest Classifier",
        "features": FEATURES,
        "version": model_data.get("version", "1.0") if clf else None,
        "classes": ["False Positive (0)", "True Positive (1)"],
        "author": "Anvesh Raju Vishwaraju"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["severity", "source_ip_reputation", "alert_frequency",
                 "bytes_transferred", "hour_of_day", "is_admin_account",
                 "failed_logins_last_hour", "alert_type_encoded"]

    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = engineer_features(data)
        pred = int(clf.predict(X)[0])
        prob = float(clf.predict_proba(X)[0][pred])

        result = format_prediction(pred, prob)
        result["input"] = data
        result["timestamp"] = datetime.now().isoformat()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if clf is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list of alert objects"}), 400

    results = []
    for i, alert in enumerate(data):
        try:
            X = engineer_features(alert)
            pred = int(clf.predict(X)[0])
            prob = float(clf.predict_proba(X)[0][pred])
            result = format_prediction(pred, prob)
            result["alert_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"alert_index": i, "error": str(e)})

    true_positives = sum(1 for r in results if "TRUE POSITIVE" in r.get("prediction", ""))
    return jsonify({
        "total_alerts": len(data),
        "true_positives": true_positives,
        "false_positives": len(data) - true_positives,
        "results": results
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SOC Alert Classifier API")
    print("  Author: Anvesh Raju Vishwaraju")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
