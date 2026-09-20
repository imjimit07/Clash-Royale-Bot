"""Performance metrics (Part 4.3)."""

from __future__ import annotations


class Metrics:
    def __init__(self) -> None:
        self.state_times: dict[str, list[float]] = {}
        self.detect_times: list[float] = []

    def record_state(self, name: str, duration: float) -> None:
        self.state_times.setdefault(name, []).append(float(duration))

    def record_detect(self, duration: float) -> None:
        self.detect_times.append(float(duration))

    def report(self, logger=None) -> None:
        for name, times in self.state_times.items():
            avg = sum(times) / len(times) if times else 0.0
            msg = f"State {name}: avg {avg:.2f}s over {len(times)} runs"
            if logger is not None:
                try:
                    logger.log(msg)
                    continue
                except Exception:
                    pass
            print(msg)
