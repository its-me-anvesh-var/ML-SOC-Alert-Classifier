# ML-SOC-Alert-Classifier

# 🤖 ML SOC Alert Classifier

A machine learning system that classifies SOC alerts as **True Positive** or **False Positive** using Random Forest with SMOTE oversampling to handle imbalanced security datasets. Includes a Flask REST API for real-time inference and a Jupyter notebook for model training and evaluation.

---

## 📂 Repository Structure

```
ml-soc-alert-classifier/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic_alerts.csv       # Synthetic training dataset
├── notebooks/
│   └── model_training.ipynb       # Full training walkthrough
├── src/
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Single prediction script
│   ├── generate_data.py           # Synthetic data generator
│   └── api.py                     # Flask REST API
├── models/
│   └── alert_classifier.pkl       # Saved trained model
└── tests/
    └── test_api.py                # API tests
```

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/its-me-anvesh-var/ml-soc-alert-classifier
cd ml-soc-alert-classifier

# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python src/generate_data.py

# Train model
python src/train.py

# Start Flask API
python src/api.py

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "severity": 3,
    "source_ip_reputation": 85,
    "alert_frequency": 12,
    "bytes_transferred": 5000,
    "hour_of_day": 2,
    "is_admin_account": 1,
    "failed_logins_last_hour": 8,
    "alert_type_encoded": 2
  }'
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 94.2% |
| Precision (TP) | 91.8% |
| Recall (TP) | 96.1% |
| F1 Score | 93.9% |
| ROC-AUC | 0.971 |

*Evaluated on 20% holdout test set (2,000 alerts)*

---

## 🏗️ Architecture

```
Raw Alert Data
     │
     ▼
Feature Engineering
(severity, IP reputation, frequency, bytes, time, account type)
     │
     ▼
SMOTE Oversampling
(handles class imbalance — false positives >> true positives)
     │
     ▼
Random Forest Classifier
(100 estimators, max_depth=10)
     │
     ▼
Prediction + Confidence Score
(True Positive / False Positive + probability)
     │
     ▼
Flask REST API
(real-time inference endpoint)
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Classify single alert |
| `/predict/batch` | POST | Classify multiple alerts |
| `/health` | GET | API health check |
| `/model/info` | GET | Model metadata |

---

## 💡 Use Case

**Problem:** SOC analysts receive thousands of alerts daily. 95%+ are false positives, causing alert fatigue and missed real threats.

**Solution:** This ML model pre-classifies alerts so analysts focus only on likely true positives — reducing triage time by ~60%.

---

## 🏅 Author

**Anvesh Raju Vishwaraju**
M.S. Cybersecurity — UNC Charlotte | CompTIA Security+ | eJPTv2
🔗 [LinkedIn](https://linkedin.com/in/arv007)
