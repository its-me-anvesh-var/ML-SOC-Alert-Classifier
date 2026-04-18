#!/usr/bin/env python3
"""
============================================================
predict.py — Single Alert Prediction (CLI)
Author: Anvesh Raju Vishwaraju
Usage: python src/predict.py --severity 3 --ip_rep 85 ...
============================================================
"""
import pickle
import numpy as np
import pandas as pd
import argparse

MODEL_PATH = "models/alert_classifier.pkl"

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("[!] Model not found. Run: python src/train.py first.")
        exit(1)

def engineer_features(data: dict, features: list) -> pd.DataFrame:
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
    return df[features]

def main():
    parser = argparse.ArgumentParser(
        description="SOC Alert Classifier — Single Prediction"
    )
    parser.add_argument("--severity",           type=int,   required=True, help="1-4")
    parser.add_argument("--ip_rep",             type=int,   required=True, help="0-100")
    parser.add_argument("--alert_freq",         type=int,   required=True, help="Alert frequency last hour")
    parser.add_argument("--bytes",              type=int,   default=0,     help="Bytes transferred")
    parser.add_argument("--hour",               type=int,   required=True, help="Hour of day 0-23")
    parser.add_argument("--is_admin",           type=int,   default=0,     help="1 if admin account")
    parser.add_argument("--failed_logins",      type=int,   default=0,     help="Failed logins last hour")
    parser.add_argument("--alert_type",         type=int,   default=0,     help="0=Network,1=Endpoint,2=Auth,3=Malware,4=Exfil")
    args = parser.parse_args()

    model_data = load_model()
    clf      = model_data["model"]
    features = model_data["features"]

    data = {
        "severity":                args.severity,
        "source_ip_reputation":    args.ip_rep,
        "alert_frequency":         args.alert_freq,
        "bytes_transferred":       args.bytes,
        "hour_of_day":             args.hour,
        "is_admin_account":        args.is_admin,
        "failed_logins_last_hour": args.failed_logins,
        "alert_type_encoded":      args.alert_type,
    }

    X    = engineer_features(data, features)
    pred = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][pred])
    conf = round(prob * 100, 2)

    print(f"\n{'='*45}")
    print(f"  SOC ALERT PREDICTION")
    print(f"{'='*45}")
    print(f"  Result:     {'🔴 TRUE POSITIVE' if pred==1 else '🟢 FALSE POSITIVE'}")
    print(f"  Confidence: {conf}%")
    print(f"  Priority:   {'P1-CRITICAL' if pred==1 and conf>=90 else 'P2-HIGH' if pred==1 else 'AUTO-CLOSE'}")
    print(f"{'='*45}\n")

if __name__ == "__main__":
    main()
