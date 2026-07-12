"""Data-source abstraction.

Every data source (the built-in synthetic/local one, or a future Google Search
Console / GA4 / Shopify adapter) implements the same :class:`DataSource`
interface. Its job is to populate the three standard CSVs in ``datasets/`` so
the rest of the pipeline — models, analysis, agent — works unchanged regardless
of where the data came from.

This is the seam that lets GrowthMind graduate from synthetic data to live
business data without touching a single model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract source that can sync business data into ``datasets/``."""

    #: Short identifier used by the connector registry / CLI (``--source``).
    name: str = "base"

    @abstractmethod
    def sync(self) -> dict:
        """Populate ``datasets/`` and return a small summary dict.

        Implementations must ensure ``daily_metrics.csv``, ``pages.csv`` and
        ``users.csv`` exist and are current after this call. The summary is
        surfaced in agent reports (e.g. row counts, date range, source name).
        """

    def is_available(self) -> bool:
        """Whether this source can run right now (e.g. credentials present)."""
        return True
