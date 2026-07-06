"""
Feature extraction from raw CAN messages stored in SQLite.

Features per message:
  id_freq         — frequency of this CAN ID in a sliding 100-msg window
  iat             — inter-arrival time from previous message (seconds)
  payload_entropy — Shannon entropy of the 8 payload bytes
  id_transition   — hash of (prev_can_id, cur_can_id) bigram
  byte0..byte7    — raw payload bytes
"""

import math
import time
import sqlite3
import numpy as np
from pipeline.db import init_db, insert_feature_batch, get_conn, DB_PATH

WINDOW = 100  # messages for frequency count


def _entropy(payload_hex: str) -> float:
    data = bytes.fromhex(payload_hex)
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def _id_transition_hash(prev: str, cur: str) -> int:
    # Stable hash of the (prev, cur) bigram that fits in a signed int
    return hash((prev, cur)) & 0x7FFFFFFF


def extract(batch_size: int = 50_000):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0]
    print(f"  Extracting features from {total:,} messages...")

    # Load all raw messages ordered by timestamp
    # (they're already sorted, but explicit ORDER BY for safety)
    cursor = conn.execute(
        "SELECT id, timestamp, can_id, dlc, payload, label "
        "FROM raw_messages ORDER BY timestamp, id"
    )

    id_window: list[str] = []
    prev_ts: float = None
    prev_id: str = None
    feat_rows: list[dict] = []
    processed = 0
    t0 = time.time()

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for row in rows:
            msg_id, ts, can_id, dlc, payload, label = (
                row["id"], row["timestamp"], row["can_id"],
                row["dlc"], row["payload"], row["label"],
            )

            # IAT
            iat = (ts - prev_ts) if prev_ts is not None else 0.0

            # ID frequency in sliding window
            id_window.append(can_id)
            if len(id_window) > WINDOW:
                id_window.pop(0)
            id_freq = id_window.count(can_id) / len(id_window)

            # Payload bytes
            raw = bytes.fromhex(payload)
            # pad/truncate to 8 bytes (should always be 8 from generator)
            raw = (raw + b'\x00' * 8)[:8]

            feat_rows.append({
                "msg_id": msg_id,
                "id_freq": id_freq,
                "iat": iat,
                "payload_entropy": _entropy(payload),
                "id_transition": _id_transition_hash(prev_id or "0000", can_id),
                "byte0": raw[0], "byte1": raw[1], "byte2": raw[2], "byte3": raw[3],
                "byte4": raw[4], "byte5": raw[5], "byte6": raw[6], "byte7": raw[7],
                "label": label,
            })

            prev_ts = ts
            prev_id = can_id

        insert_feature_batch(conn, feat_rows)
        conn.commit()
        processed += len(feat_rows)
        feat_rows.clear()
        print(f"    {processed:,}/{total:,} ({processed/total*100:.1f}%)", end="\r")

    elapsed = time.time() - t0
    print(f"\n  Done — {processed:,} features in {elapsed:.1f}s")
    conn.close()


def load_features_as_arrays(conn: sqlite3.Connection):
    """Return (X, y, ids) numpy arrays for model training."""
    rows = conn.execute(
        "SELECT id_freq, iat, payload_entropy, id_transition, "
        "byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7, label "
        "FROM features ORDER BY id"
    ).fetchall()

    X = np.array([[
        r["id_freq"], r["iat"], r["payload_entropy"], r["id_transition"],
        r["byte0"], r["byte1"], r["byte2"], r["byte3"],
        r["byte4"], r["byte5"], r["byte6"], r["byte7"],
    ] for r in rows], dtype=np.float32)

    labels = [r["label"] for r in rows]
    return X, labels


if __name__ == "__main__":
    init_db()
    extract()
