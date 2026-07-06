"""End-to-end smoke check for SentryCAN."""
import requests
import sys

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")

# --- API health ---
try:
    r = requests.get("http://localhost:8000/health", timeout=3)
    h = r.json()
    check("API /health 200",     r.status_code == 200)
    check("API model=cnn",       h.get("model") == "cnn")
except Exception as e:
    check("API reachable", False, str(e))

# --- /predict: single normal frame ---
try:
    r = requests.post("http://localhost:8000/predict", json={
        "can_id": "0329", "dlc": 8,
        "payload": "0050005000500050",
        "timestamp": 1700000100.0,
    }, timeout=5)
    p = r.json()
    check("/predict returns label",      "label" in p)
    check("/predict returns confidence", "confidence" in p)
    check("/predict returns latency_ms", "latency_ms" in p)
    check("/predict latency < 500ms",    p.get("latency_ms", 999) < 500)
except Exception as e:
    check("/predict reachable", False, str(e))

# --- /predict: DoS burst (need context) ---
sim_t = 1700010000.0
# seed 60 normal frames
for i in range(60):
    sim_t += 0.002
    requests.post("http://localhost:8000/predict", json={
        "can_id": ["0260","0316","0329","03A0"][i%4], "dlc": 8,
        "payload": "0050005000500050", "timestamp": sim_t,
    }, timeout=5)
# now burst 50 DoS frames
for _ in range(50):
    sim_t += 0.0001
    r = requests.post("http://localhost:8000/predict", json={
        "can_id": "0244", "dlc": 8,
        "payload": "FFFFFFFFFFFFFFFF", "timestamp": sim_t,
    }, timeout=5)
dos_label = r.json().get("label")
check("DoS burst detected", dos_label == "DoS", f"got {dos_label}")

# --- /stats ---
try:
    r = requests.get("http://localhost:8000/stats", timeout=3)
    s = r.json()
    check("/stats total > 0",   s.get("total", 0) > 0)
    check("/stats has attacks", "attacks" in s)
except Exception as e:
    check("/stats reachable", False, str(e))

# --- Dashboard reachable ---
try:
    r = requests.get("http://localhost:8501", timeout=5)
    check("Dashboard HTTP 200", r.status_code == 200)
except Exception as e:
    check("Dashboard reachable", False, str(e))

# --- Summary ---
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("Failed:", FAIL)
    sys.exit(1)
else:
    print("All checks passed.")
