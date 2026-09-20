"""Crash dumps + debug screenshots (Part 4.2 / 4.4)."""

from __future__ import annotations

import json
import time
import traceback


def save_debug_screenshot(frame, tag: str = "unknown"):
    try:
        from clbot.detection.image_rec import save_debug_screenshot as _save

        return _save(frame, tag)
    except Exception:
        return None


def dump_crash(logger, state, history):
    try:
        import os

        os.makedirs("debug_screens", exist_ok=True)
        path = f"debug_screens/crash_{int(time.time())}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"state": state, "history": history, "traceback": traceback.format_exc()},
                f,
                indent=2,
                default=str,
            )
        try:
            logger.error(f"Crash dump written to {path}")
        except Exception:
            pass
        return path
    except Exception:
        return None
