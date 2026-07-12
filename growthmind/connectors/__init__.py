"""Pluggable data-source connectors for GrowthMind AI."""

from __future__ import annotations

from .base import DataSource
from .google import Analytics4Connector, SearchConsoleConnector
from .local import LocalConnector

# Registry keyed by connector name.
_REGISTRY = {
    LocalConnector.name: LocalConnector,
    SearchConsoleConnector.name: SearchConsoleConnector,
    Analytics4Connector.name: Analytics4Connector,
}


def available_sources() -> list[str]:
    return sorted(_REGISTRY)


def get_connector(name: str = "local", **kwargs) -> DataSource:
    """Instantiate a connector by name (default: the local synthetic source)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown source '{name}'. Available: {', '.join(available_sources())}"
        )
    return cls(**kwargs)


__all__ = [
    "DataSource",
    "LocalConnector",
    "SearchConsoleConnector",
    "Analytics4Connector",
    "get_connector",
    "available_sources",
]
