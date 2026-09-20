"""Central config.yaml loader (Part 5.2)."""

from __future__ import annotations

from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config; returns {} when PyYAML or file is unavailable."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
