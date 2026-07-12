"""Synthetic network-traffic generator.

The project ships without a bundled dataset so it can run fully offline. This
module fabricates a realistic-looking flow dataset that mixes benign traffic
with several well-known intrusion patterns (DoS floods, port scans and
brute-force login attempts). The generated feature distributions are separable
enough to be interesting yet noisy enough to be non-trivial for an anomaly
detector.

Swap this out for a loader over a real corpus (e.g. NSL-KDD, CIC-IDS-2017)
and the rest of the pipeline works unchanged as long as the column schema in
``config.NUMERIC_FEATURES`` is preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_COLUMN, NUMERIC_FEATURES, RANDOM_STATE


def _normal_traffic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Benign flows: short-lived, balanced, low failure rates."""
    duration = rng.exponential(scale=2.0, size=n)
    src_bytes = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    dst_bytes = rng.lognormal(mean=6.5, sigma=1.0, size=n)
    packet_count = rng.poisson(lam=20, size=n) + 1
    packet_rate = packet_count / np.clip(duration, 0.05, None)
    byte_rate = (src_bytes + dst_bytes) / np.clip(duration, 0.05, None)
    failed_logins = rng.binomial(n=1, p=0.02, size=n)
    num_connections = rng.poisson(lam=3, size=n) + 1
    syn_ratio = rng.beta(a=2, b=20, size=n)
    unique_ports = rng.poisson(lam=2, size=n) + 1
    return _assemble(
        duration, src_bytes, dst_bytes, packet_count, packet_rate, byte_rate,
        failed_logins, num_connections, syn_ratio, unique_ports, label=0,
    )


def _dos_traffic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Denial-of-service floods: huge packet/byte rates, high SYN ratio."""
    duration = rng.exponential(scale=0.5, size=n)
    src_bytes = rng.lognormal(mean=9.0, sigma=1.2, size=n)
    dst_bytes = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    packet_count = rng.poisson(lam=800, size=n) + 100
    packet_rate = packet_count / np.clip(duration, 0.01, None)
    byte_rate = (src_bytes + dst_bytes) / np.clip(duration, 0.01, None)
    failed_logins = rng.binomial(n=1, p=0.01, size=n)
    num_connections = rng.poisson(lam=200, size=n) + 50
    syn_ratio = rng.beta(a=20, b=2, size=n)
    unique_ports = rng.poisson(lam=1, size=n) + 1
    return _assemble(
        duration, src_bytes, dst_bytes, packet_count, packet_rate, byte_rate,
        failed_logins, num_connections, syn_ratio, unique_ports, label=1,
    )


def _portscan_traffic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Port scans: many distinct ports, tiny payloads, many connections."""
    duration = rng.exponential(scale=1.0, size=n)
    src_bytes = rng.lognormal(mean=3.0, sigma=0.8, size=n)
    dst_bytes = rng.lognormal(mean=2.5, sigma=0.8, size=n)
    packet_count = rng.poisson(lam=5, size=n) + 1
    packet_rate = packet_count / np.clip(duration, 0.05, None)
    byte_rate = (src_bytes + dst_bytes) / np.clip(duration, 0.05, None)
    failed_logins = rng.binomial(n=1, p=0.03, size=n)
    num_connections = rng.poisson(lam=150, size=n) + 30
    syn_ratio = rng.beta(a=8, b=4, size=n)
    unique_ports = rng.poisson(lam=120, size=n) + 20
    return _assemble(
        duration, src_bytes, dst_bytes, packet_count, packet_rate, byte_rate,
        failed_logins, num_connections, syn_ratio, unique_ports, label=1,
    )


def _bruteforce_traffic(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Brute-force logins: many failed authentications, repeated connections."""
    duration = rng.exponential(scale=4.0, size=n)
    src_bytes = rng.lognormal(mean=5.0, sigma=0.8, size=n)
    dst_bytes = rng.lognormal(mean=5.0, sigma=0.8, size=n)
    packet_count = rng.poisson(lam=40, size=n) + 5
    packet_rate = packet_count / np.clip(duration, 0.05, None)
    byte_rate = (src_bytes + dst_bytes) / np.clip(duration, 0.05, None)
    failed_logins = rng.poisson(lam=15, size=n) + 3
    num_connections = rng.poisson(lam=50, size=n) + 10
    syn_ratio = rng.beta(a=3, b=10, size=n)
    unique_ports = rng.poisson(lam=1, size=n) + 1
    return _assemble(
        duration, src_bytes, dst_bytes, packet_count, packet_rate, byte_rate,
        failed_logins, num_connections, syn_ratio, unique_ports, label=1,
    )


def _assemble(*columns, label: int) -> pd.DataFrame:
    """Stack the ten per-feature arrays into a labelled DataFrame."""
    data = dict(zip(NUMERIC_FEATURES, columns))
    frame = pd.DataFrame(data)
    frame[LABEL_COLUMN] = label
    return frame


def generate_dataset(
    n_samples: int = 10_000,
    attack_ratio: float = 0.08,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a shuffled dataset of benign and malicious flows.

    Parameters
    ----------
    n_samples:
        Total number of flow records to produce.
    attack_ratio:
        Fraction of records that are attacks. The attacks are split evenly
        across the three attack families.
    random_state:
        Seed for reproducibility.
    """
    if not 0.0 <= attack_ratio < 1.0:
        raise ValueError("attack_ratio must be in [0, 1)")

    rng = np.random.default_rng(random_state)

    n_attack = int(round(n_samples * attack_ratio))
    n_normal = n_samples - n_attack

    # Split the attack budget across the three families.
    per_family = n_attack // 3
    remainder = n_attack - per_family * 3

    frames = [
        _normal_traffic(n_normal, rng),
        _dos_traffic(per_family + remainder, rng),
        _portscan_traffic(per_family, rng),
        _bruteforce_traffic(per_family, rng),
    ]

    dataset = pd.concat(frames, ignore_index=True)
    # Shuffle so attacks are not clustered at the tail.
    dataset = dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return dataset


if __name__ == "__main__":
    from .config import DATASET_PATH

    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    attacks = int(df[LABEL_COLUMN].sum())
    print(f"Wrote {len(df):,} flows to {DATASET_PATH} "
          f"({attacks:,} attacks / {len(df) - attacks:,} normal)")
