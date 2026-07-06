import requests
import random

BASE = "http://localhost:8000"
sim_time = 1700000000.0  # Virtual clock

def send(can_id, payload, true_label, iat=0.002):
    global sim_time
    sim_time += iat
    r = requests.post(f"{BASE}/predict", json={
        "can_id": can_id, "dlc": 8,
        "payload": payload,
        "timestamp": sim_time,
        "true_label": true_label,
    }, timeout=5)
    return r.json()

# Establish normal context first (50 frames)
NORMAL_IDS = ["0244","0260","0316","0329","03A0","03B0","043F","0545"]
print("Seeding 50 normal frames...")
for i in range(50):
    cid = NORMAL_IDS[i % len(NORMAL_IDS)]
    send(cid, "0050005000500050", "Normal", iat=0.002)

print(f"\n{'Scenario':<20} {'True':8} {'Pred':8} {'Conf':6} {'Lat':8}")
print("-" * 54)

def check(scenario, can_id, payload, true_label, n_lead=0, iat=0.002):
    # optionally send leading frames to build context
    for _ in range(n_lead):
        send(can_id, payload, true_label, iat=iat)
    r = send(can_id, payload, true_label, iat=iat)
    match = "OK" if r["label"] == true_label else "MISS"
    print(f"{scenario:<20} {true_label:8} {r['label']:8} {r['confidence']:.3f}  {r['latency_ms']:6.1f}ms  {match}")

# DoS: need burst of same ID at high frequency (0.1ms IAT)
check("DoS burst",        "0244", "FFFFFFFFFFFFFFFF", "DoS",  n_lead=40, iat=0.0001)

# Normal after DoS (context recovers after some normal frames with 2ms IAT)
for _ in range(10):
    send("0329", "0050005000500050", "Normal", iat=0.002)
check("Normal (post-DoS)", "0260", "000009C400000000", "Normal", iat=0.002)

# Fuzzy: random IDs with random payloads in sequence at 1ms IAT
for _ in range(15):
    p = "".join(f"{random.randint(0,255):02X}" for _ in range(8))
    send(f"{random.randint(0x100,0x6FF):04X}", p, "Fuzzy", iat=0.001)
p = "".join(f"{random.randint(0,255):02X}" for _ in range(8))
check("Fuzzy burst",       f"{random.randint(0x100,0x6FF):04X}", p, "Fuzzy", iat=0.001)

# Gear spoof: repeated injections on gear ID at 5ms IAT
check("Gear spoof",       "0316", "07FFFFFFFFFFFFFF", "Gear", n_lead=20, iat=0.005)

# RPM spoof: repeated injections on RPM ID at 5ms IAT
check("RPM spoof",        "0260", "00001F4000000000", "RPM",  n_lead=20, iat=0.005)

# Back to normal (2ms IAT)
for _ in range(20):
    send("0329", "0050005000500050", "Normal", iat=0.002)
check("Normal (clean)",   "0545", "0050005000500050", "Normal", iat=0.002)

print()
print("stats:", requests.get(f"{BASE}/stats").json())

