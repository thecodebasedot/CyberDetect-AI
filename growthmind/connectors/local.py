"""Local / synthetic data source.

Uses whatever CSVs already exist in ``datasets/``; if they are missing it
generates the synthetic dataset. This is the default source and makes the whole
system runnable offline with zero configuration.
"""

from __future__ import annotations

import pandas as pd

from ..config import DAILY_METRICS_CSV, PAGES_CSV, USERS_CSV
from ..data import generate_all
from .base import DataSource


class LocalConnector(DataSource):
    """Reads the local ``datasets/`` CSVs, generating them on first run."""

    name = "local"

    def __init__(self, regenerate: bool = False):
        self.regenerate = regenerate

    def sync(self) -> dict:
        exists = DAILY_METRICS_CSV.exists() and PAGES_CSV.exists() and USERS_CSV.exists()
        if self.regenerate or not exists:
            daily, pages, users = generate_all(save=True)
            generated = True
        else:
            daily = pd.read_csv(DAILY_METRICS_CSV, parse_dates=["date"])
            pages = pd.read_csv(PAGES_CSV)
            users = pd.read_csv(USERS_CSV)
            generated = False

        return {
            "source": self.name,
            "generated": generated,
            "daily_rows": int(len(daily)),
            "pages_rows": int(len(pages)),
            "users_rows": int(len(users)),
            "date_range": [str(daily["date"].min())[:10], str(daily["date"].max())[:10]],
        }
