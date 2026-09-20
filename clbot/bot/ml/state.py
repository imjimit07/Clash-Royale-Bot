"""Structured battle-state representation used by learned policies.

This module deliberately contains no emulator I/O.  The live bot converts its
existing detections into :class:`GameState`; training code can build the same
object from recorded JSON without an emulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class HandCard:
    """One visible card in a hand slot."""

    index: int
    name: str
    confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class EnemyThreat:
    """Coarse enemy-threat information available without object detection."""

    count: int = 0
    lane: str = "none"
    x: float = 0.5
    y: float = 0.5


@dataclass(slots=True)
class GameState:
    """Fixed-schema state consumed by the feed-forward / recurrent policies."""

    elapsed_s: float
    elixir: float
    hand: tuple[HandCard, ...] = field(default_factory=tuple)
    enemy_threat: EnemyThreat = field(default_factory=EnemyThreat)
    target_lane: str = "left"
    own_left_tower_hp: float | None = None
    own_right_tower_hp: float | None = None
    enemy_left_tower_hp: float | None = None
    enemy_right_tower_hp: float | None = None
    recent_cards: tuple[str, ...] = field(default_factory=tuple)
    source: str = "live"

    def sanitized(self) -> "GameState":
        """Return a bounded copy suitable for model features."""
        hand = tuple(
            HandCard(
                index=max(0, min(3, int(card.index))),
                name=str(card.name),
                confidence=max(0.0, min(1.0, float(card.confidence))),
            )
            for card in self.hand
        )
        threat = self.enemy_threat
        lane = threat.lane if threat.lane in {"left", "right", "none"} else "none"
        target = self.target_lane if self.target_lane in {"left", "right"} else "left"
        return GameState(
            elapsed_s=max(0.0, min(360.0, float(self.elapsed_s))),
            elixir=max(0.0, min(10.0, float(self.elixir))),
            hand=hand,
            enemy_threat=EnemyThreat(
                count=max(0, min(20, int(threat.count))),
                lane=lane,
                x=max(0.0, min(1.0, float(threat.x))),
                y=max(0.0, min(1.0, float(threat.y))),
            ),
            target_lane=target,
            own_left_tower_hp=_clip_optional(self.own_left_tower_hp),
            own_right_tower_hp=_clip_optional(self.own_right_tower_hp),
            enemy_left_tower_hp=_clip_optional(self.enemy_left_tower_hp),
            enemy_right_tower_hp=_clip_optional(self.enemy_right_tower_hp),
            recent_cards=tuple(str(x) for x in self.recent_cards[-3:]),
            source=str(self.source),
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GameState | None":
        """Build a state from a recorder action record.

        New recordings contain a state snapshot. Older packs may only contain
        ``frame_index/card_index/x/y/elapsed_s`` and are intentionally skipped
        because they do not contain enough information for supervised training.
        """
        snapshot = record.get("state")
        if not isinstance(snapshot, dict):
            return None
        raw_hand = snapshot.get("hand", [])
        hand: list[HandCard] = []
        if isinstance(raw_hand, list):
            for item in raw_hand:
                if not isinstance(item, dict):
                    continue
                try:
                    hand.append(
                        HandCard(
                            index=int(item.get("index", -1)),
                            name=str(item.get("name", "unknown")),
                            confidence=float(item.get("confidence", 1.0)),
                        )
                    )
                except (TypeError, ValueError):
                    continue
        threat_raw = snapshot.get("enemy_threat", {})
        if not isinstance(threat_raw, dict):
            threat_raw = {}
        return cls(
            elapsed_s=float(snapshot.get("elapsed_s", record.get("elapsed_s", 0.0))),
            elixir=_optional_float(snapshot.get("elixir", 0.0)) or 0.0,
            hand=tuple(hand),
            enemy_threat=EnemyThreat(
                count=int(threat_raw.get("count", 0)),
                lane=str(threat_raw.get("lane", "none")),
                x=float(threat_raw.get("x", 0.5)),
                y=float(threat_raw.get("y", 0.5)),
            ),
            target_lane=str(snapshot.get("target_lane", "left")),
            own_left_tower_hp=_optional_float(snapshot.get("own_left_tower_hp")),
            own_right_tower_hp=_optional_float(snapshot.get("own_right_tower_hp")),
            enemy_left_tower_hp=_optional_float(snapshot.get("enemy_left_tower_hp")),
            enemy_right_tower_hp=_optional_float(snapshot.get("enemy_right_tower_hp")),
            recent_cards=tuple(str(x) for x in snapshot.get("recent_cards", [])),
            source="recording",
        ).sanitized()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))
