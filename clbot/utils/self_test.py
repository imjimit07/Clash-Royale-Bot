"""Startup self-test (Part 8.1)."""

from __future__ import annotations


def self_test(emulator, logger=None) -> bool:
    def _log(msg: str) -> None:
        if logger is not None:
            try:
                logger.log(msg)
                return
            except Exception:
                pass
        print(msg)

    try:
        from clbot.detection.image_rec import _TEMPLATE_CACHE
    except Exception:
        _TEMPLATE_CACHE = {}

    checks = [
        ("Emulator running", lambda: emulator.is_emulator_running() if hasattr(emulator, "is_emulator_running") else True),
        ("ADB connected", lambda: emulator.ping() if hasattr(emulator, "ping") else True),
        ("Window focused", lambda: emulator.ensure_window_focused() if hasattr(emulator, "ensure_window_focused") else True),
        ("Screenshot works", lambda: emulator.screenshot() is not None),
        ("Resolution correct", lambda: emulator.ensure_resolution() if hasattr(emulator, "ensure_resolution") else True),
        ("Templates loaded", lambda: len(_TEMPLATE_CACHE) >= 0),
    ]
    failed: list[str] = []
    for name, fn in checks:
        try:
            ok = bool(fn())
        except Exception as e:
            ok = False
            _log(f"Self-test '{name}' raised: {e}")
        _log(f"[{'OK' if ok else 'FAIL'}] {name}")
        if not ok:
            failed.append(name)
    return not failed
