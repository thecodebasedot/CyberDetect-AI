"""Google Search Console / Analytics 4 connectors (v2 — requires credentials).

These adapters implement the same :class:`DataSource` interface as the local
connector, so the agent and pipeline can consume live Google data with no other
changes. They are intentionally left as credential-gated stubs: wiring them up
requires OAuth credentials and the Google API client libraries, which cannot be
exercised in an offline environment.

To implement (tracked in docs/ROADMAP.md):
  * GSC: query the Search Analytics API for clicks/impressions/CTR/position per
    page & query, and the URL Inspection API for index status; map onto the
    ``pages`` and ``daily_metrics`` schemas in ``growthmind/config.py``.
  * GA4: query the Data API for sessions/users/engagement/conversions/revenue;
    map onto ``daily_metrics`` and ``users``.
"""

from __future__ import annotations

import os

from .base import DataSource


class _GoogleConnectorBase(DataSource):
    """Shared credential handling for Google sources."""

    env_var = "GOOGLE_APPLICATION_CREDENTIALS"

    def is_available(self) -> bool:
        path = os.environ.get(self.env_var)
        return bool(path) and os.path.exists(path)

    def sync(self) -> dict:
        raise NotImplementedError(
            f"{self.name} connector is a v2 stub. It needs Google OAuth "
            f"credentials (set {self.env_var}) and the Google API client "
            "libraries. See docs/ROADMAP.md. Use --source local to run offline."
        )


class SearchConsoleConnector(_GoogleConnectorBase):
    name = "gsc"


class Analytics4Connector(_GoogleConnectorBase):
    name = "ga4"
