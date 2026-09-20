"""Panic hotkey (Part 8.3)."""

from __future__ import annotations


def install_panic_hotkey(shutdown_event, hotkey: str = "ctrl+shift+q") -> bool:
    """Bind a global hotkey that sets shutdown_event; False if unavailable."""
    try:
        import keyboard  # type: ignore
    except ImportError:
        return False
    try:

        def on_hotkey() -> None:
            try:
                shutdown_event.set()
            except Exception:
                pass

        keyboard.add_hotkey(hotkey, on_hotkey)
        return True
    except Exception:
        return False
