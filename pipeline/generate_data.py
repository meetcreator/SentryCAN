"""
Synthetic CAN traffic generator mimicking the HCRL Car-Hacking dataset structure.

NOTE: The HCRL dataset requires a manual browser download from Korea University's
lab site — direct programmatic fetch is not available. This generator produces
realistic synthetic traffic matching that dataset's schema and attack patterns.
This is clearly flagged here and in the README.

CAN frame structure: timestamp, CAN ID (hex), DLC, 8-byte payload, label
Labels: Normal, DoS, Fuzzy, Gear, RPM
"""

import numpy as np
import random
import time
from pipeline.db import init_db, insert_raw_batch, get_conn, DB_PATH

rng = np.random.default_rng(42)

# IDs loosely based on common automotive CAN IDs
NORMAL_IDS = [
    "0244", "0260", "0316", "0329", "03A0", "03B0",
    "043F", "0545", "05A0", "05E0", "0636", "0710",
]
# DoS floods a single ID at extremely high rate
DOS_ID = "0244"
# Fuzzy uses random IDs + random payload
FUZZY_IDS = [f"{rng.integers(0x001, 0x7FF):04X}" for _ in range(30)]
# Gear spoofing targets specific IDs
GEAR_ID = "0316"
RPM_ID = "0260"


def _normal_payload(can_id: str) -> bytes:
    """Deterministic-ish payload per ID with small random drift."""
    base = {
        "0244": bytes([0x00, 0x00, 0x1F, 0x40, 0x00, 0x00, 0x00, 0x00]),
        "0260": bytes([0x00, 0x00, 0x09, 0xC4, 0x00, 0x00, 0x00, 0x00]),  # ~2500 RPM
        "0316": bytes([0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),  # gear 3
        "0329": bytes([0x00, 0x50, 0x00, 0x50, 0x00, 0x50, 0x00, 0x50]),
    }.get(can_id, bytes(8))
    noise = rng.integers(-2, 3, size=8).astype(np.int16)
    result = np.clip(np.frombuffer(base, dtype=np.uint8).astype(np.int16) + noise, 0, 255)
    return bytes(result.astype(np.uint8))


def _dos_payload() -> bytes:
    return bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])


def _fuzzy_payload() -> bytes:
    return bytes(rng.integers(0, 256, size=8))


def _gear_spoof_payload() -> bytes:
    gear = rng.integers(1, 8)
    return bytes([gear, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])


def _rpm_spoof_payload() -> bytes:
    # Inject unrealistic RPM (~8000+)
    rpm = rng.integers(0x1F40, 0x4E20)
    hi, lo = (rpm >> 8) & 0xFF, rpm & 0xFF
    return bytes([0x00, 0x00, hi, lo, 0x00, 0x00, 0x00, 0x00])


def generate(
    n_normal: int = 400_000,
    n_dos: int = 40_000,
    n_fuzzy: int = 30_000,
    n_gear: int = 15_000,
    n_rpm: int = 15_000,
) -> list[dict]:
    """
    Generate a mixed stream of CAN messages and shuffle them (preserving
    rough timestamp ordering within each class, then interleaved).
    """
    records = []
    t = 1_700_000_000.0  # arbitrary epoch start

    # Normal traffic — realistic IAT ~2ms
    for _ in range(n_normal):
        cid = random.choice(NORMAL_IDS)
        payload = _normal_payload(cid)
        records.append({
            "timestamp": t,
            "can_id": cid,
            "dlc": 8,
            "payload": payload.hex().upper(),
            "label": "Normal",
        })
        t += rng.exponential(0.002)  # 2ms mean

    t_atk = 1_700_000_000.0 + rng.uniform(0, 50)

    # DoS — flood one ID at ~0.1ms intervals
    for _ in range(n_dos):
        records.append({
            "timestamp": t_atk,
            "can_id": DOS_ID,
            "dlc": 8,
            "payload": _dos_payload().hex().upper(),
            "label": "DoS",
        })
        t_atk += rng.exponential(0.0001)

    # Fuzzy
    t_atk = 1_700_000_000.0 + rng.uniform(100, 200)
    for _ in range(n_fuzzy):
        cid = random.choice(FUZZY_IDS)
        records.append({
            "timestamp": t_atk,
            "can_id": cid,
            "dlc": 8,
            "payload": _fuzzy_payload().hex().upper(),
            "label": "Fuzzy",
        })
        t_atk += rng.exponential(0.001)

    # Gear spoofing
    t_atk = 1_700_000_000.0 + rng.uniform(300, 400)
    for _ in range(n_gear):
        records.append({
            "timestamp": t_atk,
            "can_id": GEAR_ID,
            "dlc": 8,
            "payload": _gear_spoof_payload().hex().upper(),
            "label": "Gear",
        })
        t_atk += rng.exponential(0.005)

    # RPM spoofing
    t_atk = 1_700_000_000.0 + rng.uniform(500, 600)
    for _ in range(n_rpm):
        records.append({
            "timestamp": t_atk,
            "can_id": RPM_ID,
            "dlc": 8,
            "payload": _rpm_spoof_payload().hex().upper(),
            "label": "RPM",
        })
        t_atk += rng.exponential(0.005)

    # Sort by timestamp so the stream looks realistic
    records.sort(key=lambda r: r["timestamp"])
    return records


def main():
    print("Initialising database...")
    init_db()

    print("Generating synthetic CAN traffic...")
    t0 = time.time()
    records = generate()
    print(f"  Generated {len(records):,} messages in {time.time()-t0:.1f}s")

    print("Writing to SQLite...")
    t0 = time.time()
    conn = get_conn()
    batch_size = 10_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        insert_raw_batch(conn, batch)
    conn.commit()
    conn.close()
    print(f"  Done in {time.time()-t0:.1f}s")

    from collections import Counter
    counts = Counter(r["label"] for r in records)
    print("\nLabel distribution:")
    for label, n in sorted(counts.items()):
        print(f"  {label:8s}: {n:>7,}")


if __name__ == "__main__":
    main()
