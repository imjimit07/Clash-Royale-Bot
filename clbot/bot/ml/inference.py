"""Runtime wrapper around optional feed-forward and recurrent checkpoints."""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np

from clbot.bot.card_database import resolve_card
from clbot.bot.coords import PLAYABLE_PLAY_REGION_LTRB
from clbot.bot.ml.action import ActionValidator, PolicyAction, PolicyDecision
from clbot.bot.ml.features import FeatureEncoder
from clbot.bot.ml.policy import load_checkpoint, predict_logits, predict_sequence_logits
from clbot.bot.ml.state import GameState

DEFAULT_MODEL_PATH = Path("models") / "policy.pt"
DEFAULT_ENCODER_PATH = Path("models") / "features.json"


class LearnedPolicy:
    """Load a trained checkpoint and turn a :class:`GameState` into a safe action.

    Missing or invalid checkpoints are deliberately non-fatal so the live bot can
    continue using its deterministic policies.
    """

    def __init__(self, model_path: str | Path | None = None, encoder_path: str | Path | None = None) -> None:
        env_model = os.getenv("CLBOT_ML_MODEL")
        if model_path is not None:
            selected_model = Path(model_path)
        elif env_model:
            selected_model = Path(env_model)
        else:
            gru_default = Path("models") / "policy_gru.pt"
            selected_model = gru_default if gru_default.exists() else DEFAULT_MODEL_PATH
        self.model_path = selected_model
        self.encoder_path = Path(encoder_path or os.getenv("CLBOT_ML_ENCODER", str(DEFAULT_ENCODER_PATH)))
        self._model = None
        self._encoder: FeatureEncoder | None = None
        self._model_type = "feedforward"
        self._sequence_length = 8
        self._history: deque[np.ndarray] = deque(maxlen=self._sequence_length)
        self._failed = False
        self.validator = ActionValidator(PLAYABLE_PLAY_REGION_LTRB)

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def model_type(self) -> str:
        self._ensure_loaded()
        return self._model_type

    def reset(self) -> None:
        """Clear recurrent state at the start of a new battle."""
        self._history.clear()

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._encoder is not None:
            return True
        if self._failed or not self.model_path.exists() or not self.encoder_path.exists():
            return False
        try:
            self._encoder = FeatureEncoder.load(self.encoder_path)
            self._model, payload = load_checkpoint(self.model_path, device="cpu")
            self._model_type = str(payload.get("model_type", "feedforward"))
            self._sequence_length = max(1, int(payload.get("sequence_length") or 8))
            self._history = deque(maxlen=self._sequence_length)
            if int(payload.get("input_dim", -1)) != self._encoder.dim:
                raise ValueError("Checkpoint input_dim does not match feature encoder")
            return True
        except Exception:
            self._failed = True
            self._model = None
            self._encoder = None
            self._history.clear()
            return False

    def predict_decision(self, state: GameState) -> PolicyDecision:
        if not self._ensure_loaded():
            return PolicyDecision(None, False, 0.0, "learned checkpoint unavailable")
        assert self._model is not None and self._encoder is not None
        try:
            features = self._encoder.encode(state)
            if self._model_type == "gru":
                self._history.append(features)
                sequence = _pad_history(self._history, self._sequence_length, self._encoder.dim)
                card_logits, place_logits = predict_sequence_logits(self._model, sequence)
            else:
                self._history.clear()
                card_logits, place_logits = predict_logits(self._model, features)

            card_probs = _softmax(card_logits)
            place_probs = _softmax(place_logits)
            wait_confidence = float(card_probs[0]) if len(card_probs) else 0.0
            if wait_confidence >= self.validator.MIN_CARD_CONFIDENCE:
                return PolicyDecision(None, True, wait_confidence, "learned policy selected WAIT")

            order = np.argsort(card_probs)[::-1]
            hand_indices = {card.index for card in state.hand}
            for raw_index in order:
                model_class = int(raw_index)
                if model_class == 0:
                    continue
                card_index = model_class - 1
                if card_index not in hand_indices:
                    continue
                place_index = int(np.argmax(place_probs))
                x, y = _cell_to_coord(place_index)
                action = PolicyAction(
                    card_index=card_index,
                    x=x,
                    y=y,
                    confidence=float(card_probs[model_class]),
                    placement_confidence=float(place_probs[place_index]),
                    source="learned",
                )
                valid, reason = self.validator.validate(action, state)
                if valid:
                    return PolicyDecision(action, False, action.confidence, "learned action accepted")
                return PolicyDecision(None, False, action.confidence, f"learned action rejected: {reason}")
            return PolicyDecision(None, False, 0.0, "learned policy produced no valid card")
        except Exception as exc:
            return PolicyDecision(None, False, 0.0, f"learned inference failed: {exc}")

    def predict(self, state: GameState) -> PolicyAction | None:
        decision = self.predict_decision(state)
        return decision.action


def _pad_history(history: deque[np.ndarray], length: int, feature_dim: int) -> np.ndarray:
    values = list(history)
    if not values:
        return np.zeros((length, feature_dim), dtype=np.float32)
    while len(values) < length:
        values.insert(0, values[0])
    return np.stack(values[-length:]).astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    total = float(exp.sum())
    return (exp / total).astype(np.float32) if total > 0 else np.zeros_like(values, dtype=np.float32)


def _cell_to_coord(index: int) -> tuple[int, int]:
    cols, rows = 12, 8
    index = max(0, min(cols * rows - 1, int(index)))
    col = index % cols
    row = index // cols
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    x = int(round(left + (col + 0.5) * (right - left) / cols))
    y = int(round(top + (row + 0.5) * (bottom - top) / rows))
    return x, y


def best_action_card_cost(state: GameState, card_index: int) -> float:
    card = next(card for card in state.hand if card.index == card_index)
    cost = resolve_card(card.name).get("cost", 99)
    return float(cost) if isinstance(cost, (int, float)) else 99.0
