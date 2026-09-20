"""Robust elixir bar reader (419x633 resolution).

Measures the contiguous purple fill of the horizontal elixir bar instead of
sampling discrete pip pixels, so a single glitched pip can't freeze the bot
or fake a full bar. Bar geometry lives in :mod:`clbot.bot.coords`.
"""

from __future__ import annotations

import cv2
import numpy as np

from clbot.bot.coords import ELIXIR_BAR_X_END, ELIXIR_BAR_X_START, ELIXIR_BAR_Y


class ElixirScanner:
    """Exact 0-10 elixir count from the horizontal purple fill bar."""

    @classmethod
    def read_elixir_optional(cls, screen_frame: np.ndarray | None) -> int | None:
        """Return [0, 10], or None when the measurement itself is invalid."""
        if screen_frame is None:
            return None
        try:
            if screen_frame.ndim != 3 or screen_frame.shape[2] != 3:
                return None
            h, w = screen_frame.shape[:2]
            y1 = max(0, ELIXIR_BAR_Y - 2)
            y2 = min(h, ELIXIR_BAR_Y + 2)
            x1 = max(0, ELIXIR_BAR_X_START)
            x2 = min(w, ELIXIR_BAR_X_END)
            if y2 <= y1 or x2 <= x1:
                return None
            strip = screen_frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

            lower_purple = np.array([125, 80, 120])
            upper_purple = np.array([165, 255, 255])
            mask = cv2.inRange(hsv, lower_purple, upper_purple)

            rows = mask.shape[0]
            filled_pixels = 0
            for x in range(mask.shape[1]):
                if int(np.sum(mask[:, x])) <= 100 * rows:
                    break
                filled_pixels += 1

            total_width = x2 - x1
            if total_width <= 0:
                return None
            elixir_count = int(round(filled_pixels / total_width * 10))
            return max(0, min(10, elixir_count))
        except Exception:
            return None

    @classmethod
    def read_elixir(cls, screen_frame: np.ndarray | None) -> int:
        """Backward-compatible scanner; invalid frames return 0."""
        value = cls.read_elixir_optional(screen_frame)
        return 0 if value is None else value



class ElixirTracker:
    """Game-clock elixir model fused with bar-scanner corrections.

    The clock accrues elixir at official rates from a 5.0 battle start and
    drops on every deploy we command; the bar scanner snaps it back to
    observed reality. Either signal alone can jam — a blind scanner reads
    0 forever, a pure clock drifts from missed taps and wrong starts —
    so the estimate is fused: scanner highs apply immediately (under-
    counting starves the engine), scanner lows need three consecutive
    confirmations (one frame can glitch mid-animation).
    """

    START_ELIXIR = 5.0
    MAX_ELIXIR = 10.0
    DOWNWARD_CONFIRMATIONS = 3

    def __init__(self) -> None:
        self.estimate = self.START_ELIXIR
        self._last_update = 0.0
        self._low_streak = 0

    def start_match(self, now: float) -> None:
        """Reset the clock (call when a battle starts)."""
        self.estimate = self.START_ELIXIR
        self._last_update = now
        self._low_streak = 0

    @staticmethod
    def rate_per_second(match_elapsed_s: float) -> float:
        """Official generation rate for the current match phase."""
        if match_elapsed_s > 180.0:
            return 1.0 / 0.9
        if match_elapsed_s > 120.0:
            return 1.0 / 1.4
        return 1.0 / 2.8

    def accrue(self, now: float, match_elapsed_s: float) -> float:
        """Bank generated elixir up to the 10 cap; returns the estimate."""
        forwarded = max(0.0, now - self._last_update)
        self._last_update = now
        self.estimate = min(self.MAX_ELIXIR, self.estimate + forwarded * self.rate_per_second(match_elapsed_s))
        return self.estimate

    def deduct(self, cost: int | float) -> float:
        """Account for a commanded deploy; never allowed to raise."""
        try:
            self.estimate = max(0.0, self.estimate - float(cost))
        except (TypeError, ValueError):
            pass
        return self.estimate

    def correct(self, scanner_reading: int | float | None) -> float:
        """Fuse one valid bar-scanner reading into the estimate."""
        if scanner_reading is None:
            return self.estimate
        try:
            observed = float(scanner_reading)
        except (TypeError, ValueError):
            return self.estimate
        if observed > self.estimate:
            self.estimate = min(self.MAX_ELIXIR, observed)
            self._low_streak = 0
        elif observed < self.estimate:
            self._low_streak += 1
            if self._low_streak >= self.DOWNWARD_CONFIRMATIONS:
                self.estimate = max(0.0, observed)
                self._low_streak = 0
        else:
            self._low_streak = 0
        return self.estimate

    def read(self, frame: np.ndarray | None, now: float, match_elapsed_s: float) -> int:
        """Accrue the clock and fuse the bar scan; returns 0-10 int."""
        self.accrue(now, match_elapsed_s)
        self.correct(ElixirScanner.read_elixir_optional(frame))
        return int(self.estimate)
