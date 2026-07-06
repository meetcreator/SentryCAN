import requests
import random
import time

BASE = "http://localhost:8000"

def send(can_id, payload, true_label):
    r = requests.post(f"{BASE}/predict", json={
        "can_id": can_id, "dlc": 8,
        "payload": payload,
        "timestamp": time.time(),
        "true_label": true_label,
    }, timeout=5)
    return r.json()

# Establish normal context first (50 frames)
NORMAL_IDS = ["0244","0260","0316","0329","03A0","03B0","043F","0545"]
print("Seeding 50 normal frames...")
for i in range(50):
    cid = NORMAL_IDS[i % len(NORMAL_IDS)]
    send(cid, "0050005000500050", "Normal")

print(f"\n{'Scenario':<20} {'True':8} {'Pred':8} {'Conf':6} {'Lat':8}")
print("-" * 54)

def check(scenario, can_id, payload, true_label, n_lead=0, lead_id=None, lead_payload=None):
    # optionally send leading frames to build context
    for _ in range(n_lead):
        send(lead_id or can_id, lead_payload or payload, true_label)
    r = send(can_id, payload, true_label)
    match = "OK" if r["label"] == true_label else "MISS"
    print(f"{scenario:<20} {true_label:8} {r['label']:8} {r['confidence']:.3f}  {r['latency_ms']:6.1f}ms  {match}")

# DoS: need burst of same ID — send 40 lead frames to build id_freq
check("DoS burst",        "0244", "FFFFFFFFFFFFFFFF", "DoS",  n_lead=40)

# Normal after DoS (context should recover after a few normal frames)
for _ in range(10):
    send("0329", "0050005000500050", "Normal")
check("Normal (post-DoS)", "0260", "000009C400000000", "Normal")

# Fuzzy: random IDs with random payloads in sequence
for _ in range(15):
    p = "".join(f"{random.randint(0,255):02X}" for _ in range(8))
    send(f"{random.randint(0x100,0x6FF):04X}", p, "Fuzzy")
p = "".join(f"{random.randint(0,255):02X}" for _ in range(8))
check("Fuzzy burst",       f"{random.randint(0x100,0x6FF):04X}", p, "Fuzzy")

# Gear spoof: repeated injections on gear ID
check("Gear spoof",       "0316", "07FFFFFFFFFFFFFF", "Gear", n_lead=20)

# RPM spoof
check("RPM spoof",        "0260", "00001F4000000000", "RPM",  n_lead=20)

# Back to normal
for _ in range(20):
    send("0329", "0050005000500050", "Normal")
check("Normal (clean)",   "0545", "0050005000500050", "Normal")

print()
print("stats:", requests.get(f"{BASE}/stats").json())

