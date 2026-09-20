"""Multithreaded frame capture (Part 7.3)."""

from __future__ import annotations

import queue
import threading
import time


class FrameGrabber:
    def __init__(self, emulator, fps: int = 10) -> None:
        self.emulator = emulator
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.interval = 1.0 / max(1, fps)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                frame = self.emulator.screenshot()
            except Exception:
                frame = None
            if frame is not None:
                try:
                    self.queue.put_nowait(frame)
                except queue.Full:
                    pass
            self.stop.wait(self.interval)

    def get_frame(self, default=None):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return default

    def close(self) -> None:
        self.stop.set()
