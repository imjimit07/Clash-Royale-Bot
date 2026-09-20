"""Admin-privilege probe (Part 2.2)."""

from __future__ import annotations


def is_admin() -> bool:
    """True when running elevated; False otherwise (including non-Windows)."""
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        return bool(windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def warn_if_not_admin(log=None) -> bool:
    """Log a one-line warning when not elevated; returns is_admin()."""
    admin = is_admin()
    if not admin:
        msg = "WARNING: Run as Administrator for input/screenshot support."
        if log is not None:
            try:
                log.log(msg)
                return False
            except Exception:
                pass
        print(msg)
    return admin
