#!/usr/bin/env python3
"""
============================================================
train.py — SOC Alert Classifier Training
Author: Anvesh Raju Vishwaraju
============================================================
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

FEATURES = [
    "severity", "source_ip_reputation", "alert_frequency",
    "bytes_transferred", "hour_of_day", "is_admin_account",
    "failed_logins_last_hour", "alert_type_encoded"
]
TARGET = "label"
MODEL_PATH = "models/alert_classifier.pkl"


def load_data(path="data/synthetic_alerts.csv"):
    print(f"[*] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"    Shape: {df.shape}")
    print(f"    Class distribution:\n{df[TARGET].value_counts()}\n")
    return df


def preprocess(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Feature engineering
    X["bytes_log"] = np.log1p(X["bytes_transferred"])
    X["is_after_hours"] = ((X["hour_of_day"] < 7) |
                            (X["hour_of_day"] > 21)).astype(int)
    X["risk_score"] = (
        X["severity"] * 0.3 +
        X["source_ip_reputation"] / 100 * 0.3 +
        X["failed_logins_last_hour"] / 50 * 0.2 +
        X["is_admin_account"] * 0.2
    )

    return X, y


def train(X_train, y_train):
    print("[*] Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    Resampled: {X_res.shape[0]} samples")
    print(f"    Class distribution after SMOTE: {pd.Series(y_res).value_counts().to_dict()}\n")

    print("[*] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_res, y_res)
    return clf


def evaluate(clf, X_test, y_test):
    print("[*] Evaluating model...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print("  MODEL PERFORMANCE REPORT")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['False Positive','True Positive'])}")

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Feature importance
    features = X_test.columns.tolist()
    importances = clf.feature_importances_
    feat_imp = sorted(zip(features, importances),
                      key=lambda x: x[1], reverse=True)
    print("\n  Top Feature Importances:")
    for feat, imp in feat_imp[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {bar} {imp:.4f}")


def save_model(clf, X_train):
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": clf,
        "features": X_train.columns.tolist(),
        "version": "1.0",
        "author": "Anvesh Raju Vishwaraju"
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n[✓] Model saved to {MODEL_PATH}")


def main():
    print("\n" + "="*50)
    print("  SOC ALERT CLASSIFIER — TRAINING")
    print("  Author: Anvesh Raju Vishwaraju")
    print("="*50 + "\n")

    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Train: {len(X_train)} | Test: {len(X_test)}\n")

    clf = train(X_train, y_train)
    evaluate(clf, X_test, y_test)
    save_model(clf, X_train)

    print("\n[✓] Training complete. Run `python src/api.py` to start the API.")


if __name__ == "__main__":
    main()
