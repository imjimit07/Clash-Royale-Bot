"""Temporary diagnostic: sample BGR/HSV across the elixir-bar band of a saved frame.

Usage (PowerShell, from the repo root):
    .\\.venv\\Scripts\\python.exe scripts\\sample_hsv.py debug_screens\\<your_png>

Paste the printed table back. Delete this file after the HSV bounds are tuned.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from clbot.bot.coords import ELIXIR_BAR_X_END, ELIXIR_BAR_X_START, ELIXIR_BAR_Y


def main(path: str) -> int:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read: {path}")
        return 1
    h, w = img.shape[:2]
    print("shape:", img.shape)
    x1, x2 = max(0, ELIXIR_BAR_X_START), min(w, ELIXIR_BAR_X_END)
    for y in range(600, 632, 4):
        if y >= h:
            break
        row = img[y, x1:x2]
        print(f"y={y} mean BGR={row.mean(axis=0).astype(int)}")
    y1, y2 = max(0, ELIXIR_BAR_Y - 2), min(h, ELIXIR_BAR_Y + 2)
    strip = img[y1:y2, x1:x2]
    if strip.size == 0:
        print("Strip is empty — frame is smaller than the 419x633 playfield.")
        return 1
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    for ch, name in enumerate("HSV"):
        print(name, "min/max:", int(hsv[..., ch].min()), int(hsv[..., ch].max()))
    mask = cv2.inRange(hsv, np.array([125, 80, 120]), np.array([165, 255, 255]))
    print("in-bounds fraction:", round(float(mask.mean()) / 255, 4))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sample_hsv.py <path-to-png>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
