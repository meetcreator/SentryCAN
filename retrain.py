"""
Re-extract features (with deterministic hash) and retrain both models.
Skips data generation since raw_messages table already exists.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.db import get_conn, DB_PATH

# Clear features table so we re-extract clean
print("Clearing features table...")
conn = get_conn()
conn.execute("DELETE FROM features")
conn.commit()
conn.close()

print("[1/3] Feature extraction (deterministic hash)...")
from pipeline.features import extract
extract()

print("\n[2/3] Baseline model...")
from models.baseline import train_and_eval as train_baseline
train_baseline()

print("\n[3/3] Enhanced model (1D-CNN)...")
from models.enhanced import train_and_eval as train_cnn
train_cnn()

print("\nDone. Restart the API: uvicorn api.main:app --reload")
