"""Stable feature encoding for learned Clash Royale policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from clbot.bot.card_database import ALL_CARDS, CARD_ALIASES, lookup_card
from clbot.bot.ml.state import GameState

SCHEMA_VERSION = 3
ROLE_ORDER = (
    "WIN_CONDITION",
    "TANK",
    "SUPPORT_SPLASH",
    "SWARM",
    "BUILDING",
    "SPELL_DAMAGE",
    "SPELL_BUFF",
    "CYCLE",
    "UNKNOWN",
)


def normalize_card_name(name: str) -> str:
    """Normalize detector/OCR card ids to one stable registry id."""
    clean = str(name).lower().strip().replace(" ", "_").replace("-", "_")
    for prefix in ("evo_", "hero_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return CARD_ALIASES.get(clean, clean)


def build_vocabulary() -> list[str]:
    """Return a stable vocabulary; aliases are never separate model classes."""
    names = {normalize_card_name(name) for name in ALL_CARDS}
    names.add("unknown")
    return sorted(names)


class FeatureEncoder:
    """Convert :class:`GameState` into a deterministic fixed-width vector."""

    def __init__(self, vocabulary: Iterable[str] | None = None) -> None:
        self.vocabulary = list(vocabulary or build_vocabulary())
        if "unknown" not in self.vocabulary:
            self.vocabulary.append("unknown")
        self.card_to_index = {name: idx for idx, name in enumerate(self.vocabulary)}
        self.dim = 13 + 4 * (len(self.vocabulary) + 2 + len(ROLE_ORDER)) + 3 * len(self.vocabulary)

    def encode(self, state: GameState) -> np.ndarray:
        state = state.sanitized()
        out: list[float] = []
        out.extend(
            [
                state.elapsed_s / 360.0,
                state.elixir / 10.0,
                float(state.elapsed_s >= 120.0),
                min(state.enemy_threat.count, 10) / 10.0,
                state.enemy_threat.x,
                state.enemy_threat.y,
                float(state.enemy_threat.lane == "left"),
            ]
        )
        out.append(float(state.enemy_threat.lane == "right"))
        out.append(float(state.target_lane == "right"))

        tower_values = (
            state.own_left_tower_hp,
            state.own_right_tower_hp,
            state.enemy_left_tower_hp,
            state.enemy_right_tower_hp,
        )
        for value in tower_values:
            out.append(0.5 if value is None else max(0.0, min(1.0, value)))

        # 4 hand slots: one-hot identity + cost + role one-hot + confidence.
        hand_by_index = {card.index: card for card in state.hand}
        for slot in range(4):
            card = hand_by_index.get(slot)
            name = normalize_card_name(card.name if card else "unknown")
            idx = self.card_to_index.get(name, self.card_to_index["unknown"])
            identity = [0.0] * len(self.vocabulary)
            identity[idx] = 1.0
            out.extend(identity)
            meta = lookup_card(name) or {}
            cost = meta.get("cost", 0)
            out.append(float(cost) / 10.0 if isinstance(cost, (int, float)) else 0.0)
            role = str(meta.get("role", "UNKNOWN"))
            out.extend(float(role == expected) for expected in ROLE_ORDER)
            out.append(0.0 if card is None else max(0.0, min(1.0, card.confidence)))

        for name in state.recent_cards[-3:]:
            recent = [0.0] * len(self.vocabulary)
            idx = self.card_to_index.get(normalize_card_name(name), self.card_to_index["unknown"])
            recent[idx] = 1.0
            out.extend(recent)
        for _ in range(max(0, 3 - len(state.recent_cards))):
            out.extend([0.0] * len(self.vocabulary))

        vector = np.asarray(out, dtype=np.float32)
        if vector.size != self.dim:
            raise RuntimeError(f"Feature dimension mismatch: encoded={vector.size}, expected={self.dim}")
        return vector

    def save(self, path: str | Path) -> None:
        payload = {"schema": SCHEMA_VERSION, "vocabulary": self.vocabulary, "dim": self.dim}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature schema: {payload.get('schema')}")
        encoder = cls(payload["vocabulary"])
        if int(payload.get("dim", encoder.dim)) != encoder.dim:
            raise ValueError("Stored feature dimension does not match vocabulary")
        return encoder
