"""
FastAPI prediction endpoint for SentryCAN.

POST /predict
  body: { can_id, dlc, payload, timestamp (optional) }
  returns: { label, confidence, latency_ms, model }

GET /health  — liveness check
GET /stats   — detection log stats from SQLite

On startup, loads whichever model is available (CNN preferred, RF fallback).
"""

import os
import time
import math
import hashlib
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Silence TF logs before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "models/artifacts")
DB_PATH = os.environ.get("SENTRYCAN_DB", "data/sentrycan.db")
N_FEATURES = 12


# ---- model registry ----

class ModelRegistry:
    def __init__(self):
        self.rf = None
        self.le = None
        self.cnn = None
        self.cnn_mu = None
        self.cnn_sigma = None
        self.active = None  # 'cnn' or 'rf'

    def load(self):
        self.le = joblib.load(f"{ARTIFACT_DIR}/label_encoder.joblib")
        self.rf = joblib.load(f"{ARTIFACT_DIR}/random_forest.joblib")

        cnn_path = f"{ARTIFACT_DIR}/cnn_model.keras"
        if os.path.exists(cnn_path):
            import tensorflow as tf
            self.cnn = tf.keras.models.load_model(cnn_path)
            self.cnn_mu = np.load(f"{ARTIFACT_DIR}/cnn_scaler_mu.npy")
            self.cnn_sigma = np.load(f"{ARTIFACT_DIR}/cnn_scaler_sigma.npy")
            self.active = "cnn"
            # Warm up
            dummy = np.zeros((1, N_FEATURES, 1), dtype=np.float32)
            self.cnn(dummy, training=False)
        else:
            self.active = "rf"

        print(f"[startup] active model: {self.active}")


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    _warm_window()
    yield


app = FastAPI(title="SentryCAN API", version="0.1.0", lifespan=lifespan)


# ---- feature helpers (must match pipeline/features.py) ----

def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def _id_transition_hash(prev: str, cur: str) -> int:
    key = f"{prev}{cur}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


# Simple in-process window for frequency estimation (not shared across workers)
_id_window: list[str] = []
_prev_ts: Optional[float] = None
_prev_id: Optional[str] = None
WINDOW = 100

# Normal CAN IDs seen in training — used to warm the feature window
_NORMAL_IDS = [
    "0244", "0260", "0316", "0329", "03A0", "03B0",
    "043F", "0545", "05A0", "05E0", "0636", "0710",
]


def _warm_window():
    """Seed the sliding window with synthetic normal traffic so first
    real predictions have stable frequency estimates."""
    global _prev_ts, _prev_id
    import time as _time
    t = _time.time() - 1.0
    for i in range(WINDOW):
        cid = _NORMAL_IDS[i % len(_NORMAL_IDS)]
        _id_window.append(cid)
        _prev_ts = t
        _prev_id = cid
        t += 0.002


def extract_features(can_id: str, payload_hex: str, timestamp: float) -> np.ndarray:
    global _id_window, _prev_ts, _prev_id

    raw = bytes.fromhex(payload_hex)
    raw = (raw + b'\x00' * 8)[:8]

    iat = (timestamp - _prev_ts) if _prev_ts is not None else 0.0
    _id_window.append(can_id)
    if len(_id_window) > WINDOW:
        _id_window.pop(0)
    id_freq = _id_window.count(can_id) / len(_id_window)

    feats = np.array([
        id_freq, iat, _entropy(raw),
        _id_transition_hash(_prev_id or "0000", can_id),
        raw[0], raw[1], raw[2], raw[3],
        raw[4], raw[5], raw[6], raw[7],
    ], dtype=np.float32)

    _prev_ts = timestamp
    _prev_id = can_id
    return feats


# ---- DB helper ----

def _log_detection(row: dict):
    try:
        import sqlite3 as _sq
        conn = _sq.connect(DB_PATH)
        conn.execute(
            "INSERT INTO detection_log "
            "(timestamp, predicted_label, confidence, latency_ms, true_label, model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["timestamp"], row["predicted_label"], row["confidence"],
             row["latency_ms"], row.get("true_label"), row["model"]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # don't let logging failures break predictions


# ---- request / response ----

class CANFrame(BaseModel):
    can_id: str = Field(..., example="0244")
    dlc: int = Field(8, ge=0, le=8)
    payload: str = Field(..., example="FF00FF00FF00FF00")
    timestamp: Optional[float] = None
    true_label: Optional[str] = None  # for test-set replay evaluation


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    model: str


# ---- endpoint ----

@app.post("/predict", response_model=PredictResponse)
def predict(frame: CANFrame):
    ts = frame.timestamp or time.time()
    payload = frame.payload.upper().zfill(16)[:16]

    t0 = time.perf_counter()
    feats = extract_features(frame.can_id.upper(), payload, ts)

    if registry.active == "cnn":
        norm = (feats - registry.cnn_mu) / (registry.cnn_sigma + 1e-8)
        inp = norm.reshape(1, N_FEATURES, 1).astype(np.float32)
        proba = registry.cnn(inp, training=False).numpy()[0]
        label = registry.le.inverse_transform([int(np.argmax(proba))])[0]
        confidence = float(np.max(proba))
    else:
        proba = registry.rf.predict_proba(feats.reshape(1, -1))[0]
        label = registry.le.inverse_transform([int(np.argmax(proba))])[0]
        confidence = float(np.max(proba))

    latency_ms = (time.perf_counter() - t0) * 1000

    _log_detection({
        "timestamp": ts,
        "predicted_label": label,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "true_label": frame.true_label,
        "model": registry.active,
    })

    return PredictResponse(
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
        model=registry.active,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": registry.active}


@app.get("/stats")
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN predicted_label != 'Normal' THEN 1 ELSE 0 END) as attacks,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN true_label IS NOT NULL AND predicted_label != true_label THEN 1 ELSE 0 END) as wrong,
                SUM(CASE WHEN true_label IS NOT NULL THEN 1 ELSE 0 END) as labeled_total
            FROM detection_log
        """).fetchone()
        conn.close()
        total, attacks, avg_lat, wrong, labeled = row
        fp_rate = (wrong / labeled) if labeled else 0.0
        return {
            "total": total or 0,
            "attacks": attacks or 0,
            "avg_latency_ms": round(avg_lat or 0, 3),
            "false_positive_rate": round(fp_rate or 0, 4),
        }
    except Exception as e:
        return {"error": str(e)}
