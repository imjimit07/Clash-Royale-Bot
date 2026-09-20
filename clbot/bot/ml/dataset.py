"""Dataset extraction from recorded fight packs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from clbot.bot.coords import PLAYABLE_PLAY_REGION_LTRB
from clbot.bot.ml.features import FeatureEncoder
from clbot.bot.ml.state import GameState

DATASET_SCHEMA = 3


@dataclass(slots=True, frozen=True)
class Sample:
    game_id: str
    timestamp: float
    features: np.ndarray
    card_index: int | None
    placement_class: int
    x: int
    y: int


def iter_pack_samples(
    recordings_dir: str | Path,
    *,
    encoder: FeatureEncoder | None = None,
    policy_sources: set[str] | None = None,
) -> tuple[list[Sample], FeatureEncoder]:
    encoder = encoder or FeatureEncoder()
    root = Path(recordings_dir)
    samples: list[Sample] = []
    if not root.exists():
        return samples, encoder
    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = pack / "manifest.json"
        plays_path = pack / "plays.jsonl"
        if not manifest_path.exists() or not plays_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("outcome") not in {"win", "loss"}:
            continue
        game_id = str(manifest.get("uuid") or manifest.get("slug") or pack.name)
        for line in plays_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("confirmed") is False:
                continue
            if policy_sources is not None and str(record.get("policy_source", "unknown")) not in policy_sources:
                continue
            state = GameState.from_record(record)
            if state is None:
                continue
            action_type = str(record.get("action_type", "play"))
            if action_type == "hold":
                samples.append(
                    Sample(
                        game_id=game_id,
                        timestamp=float(record.get("elapsed_s", state.elapsed_s)),
                        features=encoder.encode(state),
                        card_index=None,
                        placement_class=0,
                        x=0,
                        y=0,
                    )
                )
                continue
            try:
                card_index = int(record["card_index"])
                x = int(record["x"])
                y = int(record["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if card_index not in {card.index for card in state.hand}:
                continue
            samples.append(
                Sample(
                    game_id=game_id,
                    timestamp=float(record.get("elapsed_s", state.elapsed_s)),
                    features=encoder.encode(state),
                    card_index=card_index,
                    placement_class=coord_to_cell(x, y),
                    x=x,
                    y=y,
                )
            )
    return samples, encoder


def save_numpy_dataset(samples: Iterable[Sample], out_dir: str | Path, encoder: FeatureEncoder) -> None:
    samples = list(samples)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if samples:
        np.save(out / "features.npy", np.stack([sample.features for sample in samples]).astype(np.float32))
        np.save(out / "card_targets.npy", np.asarray([0 if sample.card_index is None else sample.card_index + 1 for sample in samples], dtype=np.int64))
        np.save(out / "placement_targets.npy", np.asarray([sample.placement_class for sample in samples], dtype=np.int64))
        (out / "games.json").write_text(json.dumps([sample.game_id for sample in samples]), encoding="utf-8")
        (out / "timestamps.json").write_text(json.dumps([sample.timestamp for sample in samples]), encoding="utf-8")
    else:
        np.save(out / "features.npy", np.empty((0, encoder.dim), dtype=np.float32))
        np.save(out / "card_targets.npy", np.empty((0,), dtype=np.int64))
        np.save(out / "placement_targets.npy", np.empty((0,), dtype=np.int64))
        (out / "games.json").write_text("[]", encoding="utf-8")
        (out / "timestamps.json").write_text("[]", encoding="utf-8")
    encoder.save(out / "features.json")
    (out / "dataset.json").write_text(json.dumps({"schema": DATASET_SCHEMA, "n_samples": len(samples)}), encoding="utf-8")


def split_indices_by_game(game_ids: Iterable[str], validation_fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split decision rows by match so no single game leaks into train and validation."""
    games = np.asarray([str(game) for game in game_ids])
    unique = np.unique(games)
    if unique.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
    val_games = set(shuffled[:n_val])
    val_mask = np.asarray([game in val_games for game in games], dtype=bool)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64)
    if len(train_idx) == 0 and len(val_idx):
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]
    return train_idx, val_idx


def coord_to_cell(x: int, y: int) -> int:
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    cols, rows = 12, 8
    x = min(max(int(x), left), right)
    y = min(max(int(y), top), bottom)
    col = min(cols - 1, max(0, int((x - left) / max(1, right - left) * cols)))
    row = min(rows - 1, max(0, int((y - top) / max(1, bottom - top) * rows)))
    return row * cols + col
