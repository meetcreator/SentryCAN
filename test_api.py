import requests
import random
import json

BASE = "http://localhost:8000"

cases = [
    {"can_id": "0244", "dlc": 8, "payload": "FFFFFFFFFFFFFFFF", "true_label": "DoS"},
    {"can_id": "0329", "dlc": 8, "payload": "0050005000500050", "true_label": "Normal"},
    {"can_id": "0260", "dlc": 8, "payload": "000009C400000000", "true_label": "Normal"},
    {"can_id": "0316", "dlc": 8, "payload": "0300000000000000", "true_label": "Normal"},
    {
        "can_id": f"{random.randint(0x100, 0x6FF):04X}",
        "dlc": 8,
        "payload": "".join(f"{random.randint(0,255):02X}" for _ in range(8)),
        "true_label": "Fuzzy",
    },
    {"can_id": "0316", "dlc": 8, "payload": "07FFFFFFFFFFFFFF", "true_label": "Gear"},
]

print(f"{'True':8s} {'Pred':8s} {'Conf':6s} {'Lat(ms)':8s}")
print("-" * 36)
for c in cases:
    r = requests.post(f"{BASE}/predict", json=c, timeout=5)
    d = r.json()
    match = "OK" if d["label"] == c["true_label"] else "MISS"
    print(f"{c['true_label']:8s} {d['label']:8s} {d['confidence']:.3f}  {d['latency_ms']:7.1f}  {match}")

print()
print("stats:", json.dumps(requests.get(f"{BASE}/stats").json(), indent=2))
