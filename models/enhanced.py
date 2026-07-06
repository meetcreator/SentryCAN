"""
Enhanced model: lightweight 1D-CNN for multi-class CAN intrusion detection.

Input: 12-feature vector reshaped to (12, 1) for 1D convolution.
Output: softmax over 5 classes (Normal, DoS, Fuzzy, Gear, RPM).

Target: < 500KB saved model, < 5ms p99 inference on CPU.
"""

import os
import json
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info/warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from pipeline.db import get_conn, init_db
from pipeline.features import load_features_as_arrays

ARTIFACT_DIR = "models/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

N_FEATURES = 12


def build_cnn(n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(N_FEATURES, 1), name="features")
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    model = keras.Model(inp, out)
    return model


def train_and_eval():
    print("Loading features from SQLite...")
    conn = get_conn()
    X, labels = load_features_as_arrays(conn)
    conn.close()

    le: LabelEncoder = joblib.load(f"{ARTIFACT_DIR}/label_encoder.joblib")
    y = le.transform(labels)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print(f"  {X.shape[0]:,} samples, classes: {class_names}")

    # Normalise features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma
    np.save(f"{ARTIFACT_DIR}/cnn_scaler_mu.npy", mu)
    np.save(f"{ARTIFACT_DIR}/cnn_scaler_sigma.npy", sigma)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    # Reshape for Conv1D
    X_train_c = X_train.reshape(-1, N_FEATURES, 1)
    X_test_c = X_test.reshape(-1, N_FEATURES, 1)
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)

    print(f"\nBuilding 1D-CNN...")
    model = build_cnn(n_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    cb = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        X_train_c, y_train_cat,
        validation_split=0.1,
        epochs=20,
        batch_size=2048,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Eval
    y_pred_proba = model(X_test_c.astype(np.float32), training=False).numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    print("\n-- 1D-CNN (multi-class) --")
    report = classification_report(y_test_labels, y_pred_labels, digits=4, output_dict=True)
    print(classification_report(y_test_labels, y_pred_labels, digits=4))

    # Latency benchmark
    single = X_test_c[:1]
    # Warm up
    for _ in range(10):
        model(single, training=False)
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        model(single, training=False)
        latencies.append((time.perf_counter() - t0) * 1000)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    print(f"\nSingle-message latency — p50: {p50:.3f}ms  p99: {p99:.3f}ms")

    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Greens", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("1D-CNN — Confusion Matrix")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix_cnn.png", dpi=150)
    plt.close(fig)

    # Save model
    model.save(f"{ARTIFACT_DIR}/cnn_model.keras")
    size_kb = os.path.getsize(f"{ARTIFACT_DIR}/cnn_model.keras") / 1024
    print(f"\nModel size: {size_kb:.1f} KB")

    metrics = {
        "cnn": {
            "accuracy": accuracy_score(y_test_labels, y_pred_labels),
            "per_class": {k: v for k, v in report.items() if k in class_names},
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
            "model_size_kb": size_kb,
        },
        "class_names": class_names,
    }
    with open("results/cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to results/cnn_metrics.json")
    return metrics


if __name__ == "__main__":
    init_db()
    train_and_eval()
