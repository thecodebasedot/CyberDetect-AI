# CyberDetect AI

> **Network Intrusion Detection powered by an Isolation Forest anomaly detector.**

CyberDetect AI is a small, self-contained machine-learning system that flags
suspicious network traffic **without needing labelled attack data**. It learns
what "normal" traffic looks like and reports flows that stand out — the way a
real network defender operates, since you can never enumerate every future
attack in advance.

The whole project runs offline: it synthesizes a realistic network-flow
dataset, trains the model, evaluates it, and scans traffic — all from a single
CLI.

---

## Why Isolation Forest?

Intrusions are **rare and diverse**. Building a supervised classifier would
require a labelled example of every attack type you ever want to catch, which
is impossible in practice. **Isolation Forest** is an *unsupervised* anomaly
detector: it repeatedly partitions the data with random splits and isolates
points that get separated in only a few splits. Anomalous flows — those far
from the dense region of normal traffic — are isolated quickly and receive a
high anomaly score.

This makes it a natural fit for intrusion detection:

- No attack labels required for training.
- Detects **novel / zero-day** patterns, not just known signatures.
- Fast to train and score, and scales to large flow volumes.

---

## Features

- **Synthetic traffic generator** that mixes benign flows with three attack
  families: **DoS floods**, **port scans**, and **brute-force logins**.
- **Isolation Forest detector** wrapped with feature scaling and one-call
  save/load.
- **Evaluation suite** (accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix) against ground-truth labels.
- **Batch scanner** that ranks flows from most to least suspicious.
- **Command-line interface** covering the full workflow.
- **Unit tests** (`pytest`).

---

## Installation

```bash
git clone https://github.com/thecodebasedot/cyberdetect-ai.git
cd cyberdetect-ai
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Quick start

Run the entire pipeline (generate → train → detect) in one command:

```bash
python main.py demo
```

Or step through it:

```bash
# 1. Generate a synthetic traffic dataset
python main.py generate --samples 20000 --attack-ratio 0.08

# 2. Train the detector and print evaluation metrics
python main.py train

# 3. Scan a CSV of flows and list the most suspicious ones
python main.py detect --input data/network_traffic.csv --top 15
```

### Example output

```
Detection performance
---------------------
  Accuracy   :  0.992
  Precision  :  0.934
  Recall     :  0.964
  F1-score   :  0.949
  ROC-AUC    :  0.999

Confusion matrix
----------------
  TN=2195   FP=13
  FN=7      TP=185
```

---

## How it works

```
 raw flows (CSV)
      │
      ▼
 preprocessing.py   → select numeric features, sanitize, StandardScaler
      │
      ▼
 model.py           → IsolationForest.fit()  (unsupervised, labels ignored)
      │
      ▼
 detect.py          → anomaly score per flow → rank → flag attacks
      │
      ▼
 evaluate.py        → accuracy / precision / recall / F1 / ROC-AUC
```

### Feature schema

Each network flow is described by ten numeric features exported by a
lightweight sensor (KDD-Cup-inspired):

| Feature           | Meaning                                        |
| ----------------- | ---------------------------------------------- |
| `duration`        | connection lifetime (seconds)                  |
| `src_bytes`       | bytes sent source → destination                |
| `dst_bytes`       | bytes sent destination → source                |
| `packet_count`    | packets in the flow                            |
| `packet_rate`     | packets per second                             |
| `byte_rate`       | bytes per second                               |
| `failed_logins`   | failed authentication attempts                 |
| `num_connections` | connections to the same host in a short window |
| `syn_ratio`       | fraction of packets that are TCP SYN           |
| `unique_ports`    | distinct destination ports touched             |

To use **real** data instead of the synthetic generator, produce a CSV with
these columns (an optional `label` column of `0`/`1` enables evaluation) and
point `detect` at it.

---

## Project layout

```
CyberDetect-AI/
├── main.py                 # CLI entry point
├── requirements.txt
├── src/
│   ├── config.py           # paths, feature schema, model hyper-parameters
│   ├── data_generator.py   # synthetic network-traffic generator
│   ├── preprocessing.py    # feature sanitizing + scaling
│   ├── model.py            # IntrusionDetector (Isolation Forest + scaler)
│   ├── train.py            # training + evaluation pipeline
│   ├── detect.py           # batch scanning / ranking
│   └── evaluate.py         # detection metrics
├── tests/
│   └── test_pipeline.py
├── data/                   # generated datasets (gitignored)
└── models/                 # trained artifacts (gitignored)
```

---

## Running the tests

```bash
pip install pytest
python -m pytest -q
```

---

## Tuning

The single most important knob is **`contamination`** in
`src/config.py` — the expected proportion of anomalies. Raise it to catch more
attacks (higher recall, more false positives); lower it to be more
conservative (higher precision, fewer alerts). Other Isolation Forest
parameters (`n_estimators`, `max_samples`, …) live in the same file.

---

## License

See [LICENSE](LICENSE).
