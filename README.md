# SentryCAN

Real-time CAN bus intrusion detection using edge AI. Built as a hackathon prototype demonstrating an end-to-end ML pipeline on automotive network data.

> **Dataset note**: The HCRL Car-Hacking Dataset (Korea University) requires a manual browser download and is not directly fetchable programmatically. This project uses a **synthetic CAN traffic generator** that matches the dataset's schema (CAN ID, DLC, 8-byte payload, timestamp) and injects the same attack categories: Normal, DoS, Fuzzy, Gear spoofing, RPM spoofing. The generator is clearly flagged as synthetic in `pipeline/generate_data.py`.

---

## Architecture

```
run_pipeline.py          Full pipeline: generate → extract → train

pipeline/
  db.py                  SQLite schema + helpers (raw_messages, features, detection_log)
  generate_data.py       Synthetic CAN traffic generator
  features.py            Feature extraction (IAT, entropy, ID freq, payload bytes)

models/
  baseline.py            Isolation Forest (anomaly) + Random Forest (multi-class)
  enhanced.py            1D-CNN (TensorFlow/Keras) — lightweight, edge-friendly
  artifacts/             Saved model files (generated at runtime)

api/
  main.py                FastAPI — POST /predict, GET /health, GET /stats

dashboard/
  app.py                 Streamlit — live replay, detection log, inject-attack button

Dockerfile.api
Dockerfile.dashboard
docker-compose.yml
```

---

## Quickstart (local)

```bash
pip install -r requirements.txt

# 1. Run full pipeline (generates data, trains models — ~5-10 min)
python run_pipeline.py

# 2. Start API (separate terminal)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Start dashboard (separate terminal)
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501  
API docs: http://localhost:8000/docs

## Docker

```bash
# Must run pipeline first to populate data/ and models/artifacts/
python run_pipeline.py

docker compose up --build
```

---

## API

```
POST /predict
Content-Type: application/json

{
  "can_id": "0244",
  "dlc": 8,
  "payload": "FF00FF00FF00FF00",
  "timestamp": 1700000000.0
}

→ { "label": "DoS", "confidence": 0.97, "latency_ms": 1.2, "model": "cnn" }
```

---

## Results (synthetic data)

See `results/summary.md` after running the pipeline.

---

## Stack

- Python, SQLite, scikit-learn, TensorFlow/Keras, FastAPI, Streamlit, Docker
