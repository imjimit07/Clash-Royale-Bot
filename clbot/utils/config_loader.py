"""Shim re-exporting the central config loader for the existing utils layout."""

from clbot.config.loader import load_config

__all__ = ["load_config"]
