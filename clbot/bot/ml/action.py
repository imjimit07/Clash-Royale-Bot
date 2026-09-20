"""Learned-action representation and live safety validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from clbot.bot.card_database import lookup_card
from clbot.bot.coords import PLAYABLE_PLAY_REGION_LTRB
from clbot.bot.ml.state import GameState


@dataclass(slots=True, frozen=True)
class PolicyAction:
    card_index: int
    x: int
    y: int
    confidence: float
    source: str = "learned"
    placement_confidence: float = 0.0


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """Model-level decision; ``hold=True`` is a deliberate no-play action."""

    action: PolicyAction | None
    hold: bool
    confidence: float
    reason: str


class ActionValidator:
    """Reject obviously unsafe learned actions before they touch the emulator."""

    MIN_CARD_CONFIDENCE = 0.55
    MIN_PLACEMENT_CONFIDENCE = 0.35

    def __init__(self, playable_ltrb: tuple[int, int, int, int] = PLAYABLE_PLAY_REGION_LTRB) -> None:
        self.left, self.top, self.right, self.bottom = playable_ltrb

    def validate(self, action: PolicyAction, state: GameState) -> tuple[bool, str]:
        state = state.sanitized()
        if action.card_index not in {card.index for card in state.hand}:
            return False, "card slot is not in the detected hand"
        card = next(card for card in state.hand if card.index == action.card_index)
        if normalize_confidence(card.confidence) < 0.5:
            return False, "selected card identity is low confidence"
        if normalize_confidence(action.confidence) < self.MIN_CARD_CONFIDENCE:
            return False, "policy confidence is too low"
        if normalize_confidence(action.placement_confidence) < self.MIN_PLACEMENT_CONFIDENCE:
            return False, "placement confidence is too low"
        meta = lookup_card(card.name)
        if meta is None:
            return False, "card identity is not in the card registry"
        cost = meta.get("cost")
        if not isinstance(cost, (int, float)):
            return False, "card has no known elixir cost"
        if state.elixir + 1e-6 < float(cost):
            return False, f"not enough elixir ({state.elixir:.1f} < {cost})"
        if not (self.left <= action.x <= self.right and self.top <= action.y <= self.bottom):
            return False, "placement is outside the playable region"
        return True, "ok"

    def sanitize_candidates(self, actions: Iterable[PolicyAction], state: GameState) -> list[PolicyAction]:
        return [action for action in actions if self.validate(action, state)[0]]


def normalize_confidence(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
