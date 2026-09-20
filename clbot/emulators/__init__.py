"""Emulator controller — Google Play Games PC (Developer Mode) only."""

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clbot.emulators.base import BaseEmulatorController


class EmulatorType(StrEnum):
    """Available emulator types with display names."""

    GOOGLE_PLAY = "Google Play Games"


def get_emulator_registry() -> dict[EmulatorType, type["BaseEmulatorController"]]:
    """Return mapping of emulator types to controller classes.

    Battle-only bot: only Google Play Games PC (Developer Mode) is supported.
    ADB primitives live in adb_base.py / adb.py and are used internally by the
    Google Play controller (default serial localhost:6520).
    """
    from clbot.emulators.google_play import GooglePlayEmulatorController

    return {EmulatorType.GOOGLE_PLAY: GooglePlayEmulatorController}


def get_available_emulators() -> list[EmulatorType]:
    """Return list of emulator types supported on current platform."""
    return [EmulatorType.GOOGLE_PLAY]
