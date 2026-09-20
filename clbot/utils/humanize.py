"""Human-like input timing (Part 3.4)."""

from __future__ import annotations

import random
import time


def human_click(emulator, x: int, y: int, jitter_px: int = 5) -> None:
    x += random.randint(-jitter_px, jitter_px)
    y += random.randint(-jitter_px, jitter_px)
    emulator.click(int(x), int(y))
    time.sleep(random.uniform(0.08, 0.25))


def human_pause(min_s: float = 0.3, max_s: float = 1.2) -> None:
    time.sleep(random.uniform(min_s, max_s))
