"""Central configuration for CyberDetect AI.

All tunable knobs (paths, feature lists, model hyper-parameters and the
schema of the synthetic network-traffic generator) live here so the rest of
the code base stays declarative.
"""

from pathlib import Path

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
DATASET_PATH = DATA_DIR / "network_traffic.csv"

# --- Feature schema -------------------------------------------------------
# A compact, KDD-Cup-inspired set of numeric flow features that a lightweight
# network sensor can realistically export per connection.
NUMERIC_FEATURES = [
    "duration",            # seconds the connection lasted
    "src_bytes",           # bytes sent from source to destination
    "dst_bytes",           # bytes sent from destination to source
    "packet_count",        # number of packets in the flow
    "packet_rate",         # packets per second
    "byte_rate",           # bytes per second
    "failed_logins",       # failed authentication attempts on the flow
    "num_connections",     # connections to the same host in a short window
    "syn_ratio",           # fraction of packets that are TCP SYN
    "unique_ports",        # distinct destination ports touched
]

LABEL_COLUMN = "label"          # 0 = normal, 1 = attack (ground truth, eval only)

# --- Model hyper-parameters ----------------------------------------------
# `contamination` is the expected proportion of anomalies in the data. It is
# the single most important knob for an Isolation Forest used as a detector.
RANDOM_STATE = 42
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": 0.08,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
