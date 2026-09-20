"""Dynamic placement + surrender logic (Part 3.3 / 3.5)."""

from __future__ import annotations

from clbot.bot.card_database import lookup_card
from clbot.bot.coords import (
    ML_CENTER_POCKET,
    ML_FALLBACK_POCKET,
    ML_KING_CENTER,
    ML_LEFT_BRIDGE,
    TACT_CENTER_PULL,
)

BEHIND_KING_TOWER = ML_KING_CENTER
CENTER_DEFENSIVE = ML_CENTER_POCKET
BRIDGE_PRESSURE = ML_LEFT_BRIDGE


def best_cluster_center(threats: list[dict]) -> tuple[int, int]:
    if not threats:
        return CENTER_DEFENSIVE
    xs = [int(t.get("pos", CENTER_DEFENSIVE)[0]) for t in threats if isinstance(t.get("pos"), (list, tuple))]
    ys = [int(t.get("pos", CENTER_DEFENSIVE)[1]) for t in threats if isinstance(t.get("pos"), (list, tuple))]
    if not xs or not ys:
        first = threats[0].get("pos", CENTER_DEFENSIVE)
        try:
            return (int(first[0]), int(first[1]))
        except Exception:
            return CENTER_DEFENSIVE
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def choose_placement(card: str, threats: list[dict], my_towers=None, elixir: float = 0) -> tuple[int, int]:
    """Role-based placement using the card registry (no raw tuples here)."""
    meta = lookup_card(card) or {}
    role = str(meta.get("role", "UNKNOWN"))
    if role in {"SPELL_DAMAGE"}:
        return best_cluster_center(threats)
    if role in {"BUILDING"}:
        return CENTER_DEFENSIVE
    if role in {"TANK", "WIN_CONDITION"} and elixir >= 8:
        return BEHIND_KING_TOWER
    if threats:
        pos = threats[0].get("pos")
        try:
            return (int(pos[0]), int(pos[1]))
        except Exception:
            return TACT_CENTER_PULL
    return BRIDGE_PRESSURE or ML_FALLBACK_POCKET


def should_surrender(state: dict) -> bool:
    my_towers = state.get("my_tower_count", 3)
    enemy_towers = state.get("enemy_tower_count", 3)
    time_left = state.get("time_left", 180)
    try:
        if int(my_towers) == 0:
            return True
        if int(my_towers) == 1 and int(enemy_towers) == 3 and float(time_left) < 30:
            return True
    except Exception:
        return False
    return False
