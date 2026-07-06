"""
End-to-end pipeline runner: generate data -> extract features -> train models.
Run this once before starting the API + dashboard.
"""

import sys
import os

# Make sure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.db import init_db
from pipeline.generate_data import main as gen_data
from pipeline.features import extract
from models.baseline import train_and_eval as train_baseline
from models.enhanced import train_and_eval as train_cnn

if __name__ == "__main__":
    print("=" * 60)
    print("SentryCAN — full pipeline")
    print("=" * 60)

    print("\n[1/4] Init DB + generate data")
    init_db()
    gen_data()

    print("\n[2/4] Feature extraction")
    extract()

    print("\n[3/4] Baseline model (Isolation Forest + Random Forest)")
    train_baseline()

    print("\n[4/4] Enhanced model (1D-CNN)")
    train_cnn()

    print("\nDone. Start services with:")
    print("  uvicorn api.main:app --reload")
    print("  streamlit run dashboard/app.py")
    print("Or: docker compose up --build")
