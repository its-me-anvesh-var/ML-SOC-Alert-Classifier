#!/usr/bin/env python3
"""
============================================================
generate_data.py — Synthetic SOC Alert Data Generator
Author: Anvesh Raju Vishwaraju
============================================================
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 10000

def generate_alerts(n=N):
    data = {
        # Alert severity (1=Low, 2=Medium, 3=High, 4=Critical)
        "severity": np.random.choice([1, 2, 3, 4], n,
                                      p=[0.3, 0.35, 0.25, 0.1]),
        # Source IP reputation score (0-100, higher = more suspicious)
        "source_ip_reputation": np.random.randint(0, 101, n),
        # How many times this alert fired in the last hour
        "alert_frequency": np.random.randint(1, 100, n),
        # Bytes transferred in the event
        "bytes_transferred": np.random.randint(0, 10_000_000, n),
        # Hour of day (0-23)
        "hour_of_day": np.random.randint(0, 24, n),
        # Is the source account an admin?
        "is_admin_account": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        # Failed login attempts in last hour
        "failed_logins_last_hour": np.random.randint(0, 50, n),
        # Alert type (0=Network, 1=Endpoint, 2=Auth, 3=Malware, 4=Exfil)
        "alert_type_encoded": np.random.choice([0, 1, 2, 3, 4], n),
    }
    df = pd.DataFrame(data)

    # Generate labels with realistic logic
    # True Positive probability based on features
    tp_score = (
        (df["severity"] >= 3).astype(int) * 0.3 +
        (df["source_ip_reputation"] > 70).astype(int) * 0.25 +
        (df["is_admin_account"] == 1).astype(int) * 0.2 +
        (df["failed_logins_last_hour"] > 10).astype(int) * 0.15 +
        ((df["hour_of_day"] < 7) | (df["hour_of_day"] > 21)).astype(int) * 0.1
    )

    # Add noise
    tp_score += np.random.normal(0, 0.1, n)
    tp_score = tp_score.clip(0, 1)

    # Label: 1 = True Positive, 0 = False Positive
    # ~15% true positive rate (realistic for SOC)
    df["label"] = (tp_score > 0.65).astype(int)

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_alerts()
    df.to_csv("data/synthetic_alerts.csv", index=False)
    print(f"Generated {len(df)} alerts")
    print(f"True Positives:  {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"False Positives: {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")
    print("Saved to data/synthetic_alerts.csv")
