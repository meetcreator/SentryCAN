"""
Streamlit dashboard — live CAN traffic replay with FastAPI backend.

Light theme, no neon, no gradients, functional labels.
Calls POST /predict for each message instead of running model inline.
"""

import time
import random
import sqlite3
import os

import requests
import streamlit as st
import pandas as pd

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DB_PATH = os.environ.get("SENTRYCAN_DB", "data/sentrycan.db")
REPLAY_BATCH = 1          # messages sent per tick
TICK_INTERVAL = 0.05      # seconds between ticks

st.set_page_config(
    page_title="SentryCAN",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- minimal light-theme CSS ----
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] {
    background: #f7f8fa;
    color: #1a1a1a;
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }
  [data-testid="stHeader"] { background: #f7f8fa; }
  h1, h2, h3 { font-weight: 600; letter-spacing: -0.3px; }
  .metric-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 16px 20px;
  }
  .log-row-normal  { color: #374151; }
  .log-row-attack  { color: #b91c1c; font-weight: 500; }
  .tag-normal { background:#d1fae5; color:#065f46;
                padding:2px 7px; border-radius:4px; font-size:0.8rem; }
  .tag-attack { background:#fee2e2; color:#991b1b;
                padding:2px 7px; border-radius:4px; font-size:0.8rem; }
  div[data-testid="stButton"] > button {
    border-radius: 6px;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)


# ---- session state ----
def _init_state():
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("log", [])
    ss.setdefault("total", 0)
    ss.setdefault("attacks", 0)
    ss.setdefault("latencies", [])
    ss.setdefault("fp_count", 0)
    ss.setdefault("labeled_count", 0)
    ss.setdefault("test_msgs", None)
    ss.setdefault("replay_idx", 0)

_init_state()


def load_test_messages(n: int = 2000) -> list[dict]:
    """Pull a stratified sample from raw_messages for replay."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Sample equally across labels for a balanced demo
        rows = conn.execute("""
            SELECT can_id, dlc, payload, timestamp, label FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) rn
                FROM raw_messages
            ) WHERE rn <= ?
            ORDER BY timestamp
        """, (n // 5,)).fetchall()
        conn.close()
        return [dict(zip(["can_id","dlc","payload","timestamp","label"], r)) for r in rows]
    except Exception:
        return []


def _attack_burst() -> list[dict]:
    """Generate a short burst of synthetic attack frames."""
    burst = []
    for _ in range(30):
        label = random.choice(["DoS", "Fuzzy", "Gear", "RPM"])
        if label == "DoS":
            burst.append({"can_id": "0244", "dlc": 8,
                          "payload": "FFFFFFFFFFFFFFFF", "label": label,
                          "timestamp": time.time()})
        elif label == "Fuzzy":
            p = "".join(f"{random.randint(0,255):02X}" for _ in range(8))
            burst.append({"can_id": f"{random.randint(1,0x7FF):04X}", "dlc": 8,
                          "payload": p, "label": label,
                          "timestamp": time.time()})
        elif label == "Gear":
            gear = random.randint(1, 7)
            burst.append({"can_id": "0316", "dlc": 8,
                          "payload": f"{gear:02X}FFFFFFFFFFFFFF", "label": label,
                          "timestamp": time.time()})
        else:  # RPM
            rpm = random.randint(0x1F40, 0x4E20)
            burst.append({"can_id": "0260", "dlc": 8,
                          "payload": f"0000{rpm:04X}00000000", "label": label,
                          "timestamp": time.time()})
    return burst


def _send(msg: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={
            "can_id": msg["can_id"],
            "dlc": msg.get("dlc", 8),
            "payload": msg["payload"],
            "timestamp": msg.get("timestamp", time.time()),
            "true_label": msg.get("label"),
        }, timeout=2)
        return r.json()
    except Exception:
        return None


def _api_ok() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


# ---- layout ----
st.markdown("## SentryCAN")
st.caption("Real-time CAN bus intrusion detection — edge AI prototype")

api_live = _api_ok()
if not api_live:
    st.error(f"API not reachable at {API_URL}. Start the API server first.")

col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 3])

with col_ctrl1:
    if st.button("Start replay", disabled=not api_live):
        if st.session_state["test_msgs"] is None:
            with st.spinner("Loading test messages from DB..."):
                st.session_state["test_msgs"] = load_test_messages(2000)
        st.session_state["running"] = True

with col_ctrl2:
    if st.button("Pause"):
        st.session_state["running"] = False

with col_ctrl3:
    if st.button("Inject attack burst", disabled=not api_live):
        burst = _attack_burst()
        if st.session_state["test_msgs"] is not None:
            # prepend burst to front of queue
            st.session_state["test_msgs"] = burst + st.session_state["test_msgs"][st.session_state["replay_idx"]:]
            st.session_state["replay_idx"] = 0
        else:
            st.session_state["test_msgs"] = burst
            st.session_state["replay_idx"] = 0
        st.session_state["running"] = True

# ---- stats row ----
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Messages processed", st.session_state["total"])
m2.metric("Attacks detected", st.session_state["attacks"])
avg_lat = (sum(st.session_state["latencies"]) / len(st.session_state["latencies"])
           if st.session_state["latencies"] else 0.0)
m3.metric("Avg latency", f"{avg_lat:.2f} ms")
fp_rate = (st.session_state["fp_count"] / st.session_state["labeled_count"]
           if st.session_state["labeled_count"] else 0.0)
m4.metric("False positive rate", f"{fp_rate:.2%}")

st.divider()

# ---- detection log ----
log_placeholder = st.empty()

def _render_log():
    rows = st.session_state["log"][-60:][::-1]  # newest first, max 60
    if not rows:
        log_placeholder.info("No messages yet. Click 'Start replay'.")
        return

    df = pd.DataFrame(rows)
    df["tag"] = df["label"].apply(
        lambda l: f'<span class="tag-normal">Normal</span>'
                  if l == "Normal"
                  else f'<span class="tag-attack">{l}</span>'
    )
    df_display = df[["time", "can_id", "payload_short", "tag", "confidence", "latency_ms"]].copy()
    df_display.columns = ["Time", "CAN ID", "Payload (first 4B)", "Label", "Confidence", "Latency (ms)"]

    log_placeholder.write(
        df_display.to_html(escape=False, index=False,
                           classes="dataframe", border=0),
        unsafe_allow_html=True,
    )

_render_log()

# ---- replay loop ----
if st.session_state["running"] and api_live and st.session_state["test_msgs"]:
    msgs = st.session_state["test_msgs"]
    idx = st.session_state["replay_idx"]

    batch = msgs[idx: idx + REPLAY_BATCH]
    if not batch:
        st.session_state["running"] = False
        st.info("Replay complete.")
    else:
        for msg in batch:
            resp = _send(msg)
            if resp:
                pred = resp.get("label", "?")
                conf = resp.get("confidence", 0.0)
                lat = resp.get("latency_ms", 0.0)
                true_l = msg.get("label", "?")

                st.session_state["total"] += 1
                st.session_state["latencies"].append(lat)
                if len(st.session_state["latencies"]) > 1000:
                    st.session_state["latencies"].pop(0)
                if pred != "Normal":
                    st.session_state["attacks"] += 1
                if true_l and true_l != "?":
                    st.session_state["labeled_count"] += 1
                    if pred != true_l:
                        st.session_state["fp_count"] += 1

                st.session_state["log"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "can_id": msg["can_id"],
                    "payload_short": msg["payload"][:8],
                    "label": pred,
                    "confidence": f"{conf:.3f}",
                    "latency_ms": f"{lat:.2f}",
                })

        st.session_state["replay_idx"] = idx + REPLAY_BATCH
        time.sleep(TICK_INTERVAL)
        st.rerun()
