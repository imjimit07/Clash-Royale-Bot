"""Heartbeat watchdog for long-running states (leaf module — no bot imports).

A Clash Royale match runs ~180s plus overtime inside a single ``1v1_fight``
state call, so a watchdog measured from state entry misfires on every
full-length battle. Instead the worker measures *progress staleness*: long
states pulse the heartbeat as they work, and only silence trips recovery.

A cooperative stop flag lets a timed-out state thread exit promptly instead
of playing cards behind the main loop's back (Python can't kill threads).
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_last_heartbeat = time.time()
_stop_requested = threading.Event()


def reset() -> None:
    """Arm the watchdog for a new state entry (also clears any stop request)."""
    global _last_heartbeat
    _stop_requested.clear()
    with _lock:
        _last_heartbeat = time.time()


def pulse() -> None:
    """Record progress from a long-running state. Never raises."""
    global _last_heartbeat
    try:
        with _lock:
            _last_heartbeat = time.time()
    except Exception:
        pass


def staleness_s() -> float:
    """Seconds since the last heartbeat."""
    with _lock:
        return time.time() - _last_heartbeat


def is_stale(timeout_s: float) -> bool:
    """True when no progress was recorded within ``timeout_s``."""
    return staleness_s() > timeout_s


def request_stop() -> None:
    """Ask the running state thread to exit at its next checkpoint."""
    _stop_requested.set()


def stop_requested() -> bool:
    """True when the watchdog (or anyone) asked the state to stop."""
    return _stop_requested.is_set()


def clear_stop() -> None:
    """Clear a stop request (done implicitly by :func:`reset`)."""
    _stop_requested.clear()
