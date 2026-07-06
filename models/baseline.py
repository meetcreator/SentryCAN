"""
Baseline: Isolation Forest for anomaly detection + a Random Forest
multi-class classifier on top.

Two-stage:
  1. IsolationForest flags anomalies (Normal vs. attack)
  2. RandomForestClassifier assigns attack sub-type (DoS/Fuzzy/Gear/RPM)

This gives us both anomaly detection metrics AND per-class classification
metrics for the demo, using only scikit-learn.
"""

import time
import json
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from pipeline.db import get_conn, init_db
from pipeline.features import load_features_as_arrays

ARTIFACT_DIR = "models/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_and_eval():
    print("Loading features from SQLite...")
    conn = get_conn()
    X, labels = load_features_as_arrays(conn)
    conn.close()
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} features")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)
    print(f"  Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Stage 1: Isolation Forest (anomaly) ----
    print("\nTraining Isolation Forest...")
    normal_idx = np.where(y_train == le.transform(["Normal"])[0])[0]
    X_normal = X_train[normal_idx]

    t0 = time.time()
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    # Evaluate on test set — IF gives -1 (anomaly) or 1 (normal)
    t0 = time.time()
    iso_preds_raw = iso.predict(X_test)
    iso_elapsed = time.time() - t0
    iso_labels_bin = np.where(iso_preds_raw == 1, "Normal", "Attack")
    y_test_bin = np.where(y_test == le.transform(["Normal"])[0], "Normal", "Attack")

    print("\n-- Isolation Forest (binary) --")
    print(classification_report(y_test_bin, iso_labels_bin, digits=4))
    print(f"Per-message latency: {iso_elapsed/len(X_test)*1000:.4f} ms")

    # ---- Stage 2: Random Forest multi-class ----
    print("\nTraining Random Forest classifier (multi-class)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    t0 = time.time()
    rf_preds = rf.predict(X_test)
    rf_elapsed = time.time() - t0
    rf_proba = rf.predict_proba(X_test)

    rf_labels = le.inverse_transform(rf_preds)
    y_test_labels = le.inverse_transform(y_test)

    print("\n-- Random Forest (multi-class) --")
    report = classification_report(y_test_labels, rf_labels, digits=4, output_dict=True)
    print(classification_report(y_test_labels, rf_labels, digits=4))
    print(f"Per-message latency: {rf_elapsed/len(X_test)*1000:.4f} ms")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test_labels, rf_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Random Forest — Confusion Matrix")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix_baseline.png", dpi=150)
    plt.close(fig)
    print("\nConfusion matrix saved to results/confusion_matrix_baseline.png")

    # ---- Latency benchmark (single-message) ----
    single = X_test[:1]
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        rf.predict(single)
        latencies.append((time.perf_counter() - t0) * 1000)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    print(f"\nSingle-message latency — p50: {p50:.3f}ms  p99: {p99:.3f}ms")

    # ---- Save models ----
    joblib.dump(iso, f"{ARTIFACT_DIR}/isolation_forest.joblib")
    joblib.dump(rf, f"{ARTIFACT_DIR}/random_forest.joblib")
    joblib.dump(le, f"{ARTIFACT_DIR}/label_encoder.joblib")

    metrics = {
        "baseline_rf": {
            "accuracy": accuracy_score(y_test_labels, rf_labels),
            "per_class": {k: v for k, v in report.items() if k in class_names},
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
        },
        "isolation_forest_binary": {
            "report": classification_report(
                y_test_bin, iso_labels_bin, output_dict=True
            ),
            "latency_ms_per_msg": iso_elapsed / len(X_test) * 1000,
        },
        "class_names": class_names,
    }
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to results/baseline_metrics.json")

    return metrics, class_names, X_test, y_test, y_test_labels, rf_preds, rf_labels


if __name__ == "__main__":
    init_db()
    train_and_eval()
