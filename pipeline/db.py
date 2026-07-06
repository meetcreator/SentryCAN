"""
SQLite schema and helpers for SentryCAN.

Tables:
  raw_messages   — every CAN frame as received (or replayed)
  features       — extracted feature vectors keyed to raw_messages
  detection_log  — inference results from the model
"""

import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("SENTRYCAN_DB", "data/sentrycan.db")


def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def db_conn(path: str = DB_PATH):
    conn = get_conn(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str = DB_PATH):
    with db_conn(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS raw_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL NOT NULL,
                can_id      TEXT NOT NULL,   -- hex string e.g. '0244'
                dlc         INTEGER NOT NULL,
                payload     TEXT NOT NULL,   -- hex string, 16 chars
                label       TEXT NOT NULL    -- Normal/DoS/Fuzzy/Gear/RPM
            );

            CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_raw_label ON raw_messages(label);

            CREATE TABLE IF NOT EXISTS features (
                id              INTEGER PRIMARY KEY,
                msg_id          INTEGER NOT NULL REFERENCES raw_messages(id),
                id_freq         REAL,    -- messages/sec for this CAN ID in last window
                iat             REAL,    -- inter-arrival time vs previous msg, seconds
                payload_entropy REAL,
                id_transition   INTEGER, -- hash of (prev_can_id, cur_can_id)
                byte0           INTEGER,
                byte1           INTEGER,
                byte2           INTEGER,
                byte3           INTEGER,
                byte4           INTEGER,
                byte5           INTEGER,
                byte6           INTEGER,
                byte7           INTEGER,
                label           TEXT
            );

            CREATE TABLE IF NOT EXISTS detection_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                msg_id          INTEGER REFERENCES raw_messages(id),
                timestamp       REAL NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence      REAL,
                latency_ms      REAL,
                true_label      TEXT,
                model           TEXT   -- 'isolation_forest' or 'cnn'
            );
        """)


def insert_raw_batch(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany(
        "INSERT INTO raw_messages (timestamp, can_id, dlc, payload, label) "
        "VALUES (:timestamp, :can_id, :dlc, :payload, :label)",
        rows,
    )


def insert_feature_batch(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany(
        """INSERT INTO features
           (msg_id, id_freq, iat, payload_entropy, id_transition,
            byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7, label)
           VALUES
           (:msg_id, :id_freq, :iat, :payload_entropy, :id_transition,
            :byte0, :byte1, :byte2, :byte3, :byte4, :byte5, :byte6, :byte7, :label)""",
        rows,
    )


def insert_detection(conn: sqlite3.Connection, row: dict):
    conn.execute(
        """INSERT INTO detection_log
           (msg_id, timestamp, predicted_label, confidence, latency_ms, true_label, model)
           VALUES
           (:msg_id, :timestamp, :predicted_label, :confidence, :latency_ms, :true_label, :model)""",
        row,
    )
    conn.commit()


def get_feature_rows(conn: sqlite3.Connection, limit: int = None):
    q = "SELECT * FROM features ORDER BY id"
    if limit:
        q += f" LIMIT {limit}"
    return conn.execute(q).fetchall()


def get_detection_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN predicted_label != 'Normal' THEN 1 ELSE 0 END) as attacks,
            AVG(latency_ms) as avg_latency,
            SUM(CASE WHEN predicted_label != true_label THEN 1 ELSE 0 END) as wrong
        FROM detection_log
    """).fetchone()
    return dict(row) if row else {}
