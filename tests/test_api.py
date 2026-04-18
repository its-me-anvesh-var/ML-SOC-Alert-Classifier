#!/usr/bin/env python3
"""
============================================================
test_api.py — API Unit Tests
Author: Anvesh Raju Vishwaraju
Usage: python tests/test_api.py
       (Start api.py first in another terminal)
============================================================
"""
import requests
import json

BASE = "http://localhost:5000"

SAMPLE_ALERT = {
    "severity": 4,
    "source_ip_reputation": 92,
    "alert_frequency": 45,
    "bytes_transferred": 8500000,
    "hour_of_day": 3,
    "is_admin_account": 1,
    "failed_logins_last_hour": 22,
    "alert_type_encoded": 2
}

LOW_RISK_ALERT = {
    "severity": 1,
    "source_ip_reputation": 10,
    "alert_frequency": 2,
    "bytes_transferred": 500,
    "hour_of_day": 10,
    "is_admin_account": 0,
    "failed_logins_last_hour": 0,
    "alert_type_encoded": 0
}

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("✅ /health — PASS")

def test_predict_high_risk():
    r = requests.post(f"{BASE}/predict", json=SAMPLE_ALERT)
    assert r.status_code == 200
    result = r.json()
    assert "prediction" in result
    assert "confidence" in result
    print(f"✅ /predict (high risk) — PASS → {result['prediction']} {result['confidence']}")

def test_predict_low_risk():
    r = requests.post(f"{BASE}/predict", json=LOW_RISK_ALERT)
    assert r.status_code == 200
    result = r.json()
    print(f"✅ /predict (low risk) — PASS → {result['prediction']} {result['confidence']}")

def test_predict_missing_field():
    bad_alert = {"severity": 3}
    r = requests.post(f"{BASE}/predict", json=bad_alert)
    assert r.status_code == 400
    assert "error" in r.json()
    print("✅ /predict (missing fields) — PASS → 400 error returned correctly")

def test_batch_predict():
    batch = [SAMPLE_ALERT, LOW_RISK_ALERT, SAMPLE_ALERT]
    r = requests.post(f"{BASE}/predict/batch", json=batch)
    assert r.status_code == 200
    result = r.json()
    assert result["total_alerts"] == 3
    assert len(result["results"]) == 3
    print(f"✅ /predict/batch — PASS → {result['true_positives']} TP / {result['false_positives']} FP")

def test_model_info():
    r = requests.get(f"{BASE}/model/info")
    assert r.status_code == 200
    assert "features" in r.json()
    print("✅ /model/info — PASS")

if __name__ == "__main__":
    print(f"\n{'='*45}")
    print("  ML SOC Classifier — API Tests")
    print("  Author: Anvesh Raju Vishwaraju")
    print(f"{'='*45}\n")
    try:
        test_health()
        test_predict_high_risk()
        test_predict_low_risk()
        test_predict_missing_field()
        test_batch_predict()
        test_model_info()
        print(f"\n{'='*45}")
        print("  ALL TESTS PASSED ✅")
        print(f"{'='*45}\n")
    except requests.exceptions.ConnectionError:
        print("\n[!] API not running. Start it first: python src/api.py\n")
    except AssertionError as e:
        print(f"\n[!] TEST FAILED: {e}\n")
