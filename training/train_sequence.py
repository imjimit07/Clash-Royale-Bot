#!/usr/bin/env python3
"""Train the GRU policy over short within-match decision sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clbot.bot.ml.dataset import split_indices_by_game
from clbot.bot.ml.features import FeatureEncoder
from clbot.bot.ml.policy import SequencePolicyNet, save_checkpoint


def _make_windows(features, cards, placements, seq_len: int):
    xs, ys_card, ys_place = [], [], []
    for end in range(seq_len - 1, len(features)):
        xs.append(features[end - seq_len + 1 : end + 1])
        ys_card.append(cards[end])
        ys_place.append(placements[end])
    return np.stack(xs), np.asarray(ys_card), np.asarray(ys_place)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path, default=Path("models/policy_gru.pt"), nargs="?")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features = np.load(args.dataset / "features.npy")
    cards = np.load(args.dataset / "card_targets.npy")
    placements = np.load(args.dataset / "placement_targets.npy")
    encoder = FeatureEncoder.load(args.dataset / "features.json")
    games = json.loads((args.dataset / "games.json").read_text(encoding="utf-8"))
    if len(features) < args.seq_len:
        raise SystemExit("Not enough samples for the requested sequence length.")

    if len(games) != len(features):
        raise SystemExit("Dataset games.json length does not match feature rows")
    train_idx, val_idx = split_indices_by_game(games)

    def make_split(indices):
        grouped: dict[str, list[int]] = {}
        for global_idx in indices:
            grouped.setdefault(str(games[int(global_idx)]), []).append(int(global_idx))
        windows, card_y, place_y = [], [], []
        for local_indices in grouped.values():
            if len(local_indices) < args.seq_len:
                continue
            x_local, c_local, p_local = _make_windows(
                features[local_indices], cards[local_indices], placements[local_indices], args.seq_len
            )
            windows.append(x_local)
            card_y.append(c_local)
            place_y.append(p_local)
        if not windows:
            return None
        return np.concatenate(windows), np.concatenate(card_y), np.concatenate(place_y)

    train_data = make_split(train_idx)
    val_data = make_split(val_idx)
    if train_data is None:
        raise SystemExit("No complete per-game training sequences found.")
    x, c, p = train_data
    print(f"train_sequences={len(x)} val_sequences={0 if val_data is None else len(val_data[0])}")
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(c), torch.from_numpy(p))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    counts = np.bincount(c, minlength=5).astype(np.float32)
    card_weights = np.ones(5, dtype=np.float32)
    nonzero = counts > 0
    card_weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    card_weights *= 5.0 / max(card_weights.sum(), 1e-6)
    model = SequencePolicyNet(encoder.dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    card_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(card_weights))
    placement_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_c, batch_p in loader:
            optimizer.zero_grad(set_to_none=True)
            card_logits, place_logits = model(batch_x)
            card_loss = card_loss_fn(card_logits, batch_c)
            play_mask = batch_c != 0
            placement_loss = placement_loss_fn(place_logits[play_mask], batch_p[play_mask]) if bool(play_mask.any()) else card_loss * 0.0
            loss = card_loss + 0.7 * placement_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
        val_loss = 0.0
        val_count = 0
        if val_data is not None:
            model.eval()
            with torch.inference_mode():
                vx, vc, vp = val_data
                vx_t = torch.from_numpy(vx.astype(np.float32))
                vc_t = torch.from_numpy(vc)
                vp_t = torch.from_numpy(vp)
                vc_logits, vp_logits = model(vx_t)
                v_card_loss = card_loss_fn(vc_logits, vc_t)
                v_play_mask = vc_t != 0
                v_place_loss = placement_loss_fn(vp_logits[v_play_mask], vp_t[v_play_mask]) if bool(v_play_mask.any()) else v_card_loss * 0.0
                val_loss = float(v_card_loss.item() + 0.7 * v_place_loss.item())
                val_count = len(vx)
        suffix = f" val_loss={val_loss:.4f}" if val_count else ""
        print(f"epoch={epoch:03d} train_loss={total/len(ds):.4f}{suffix}")

    args.model.parent.mkdir(parents=True, exist_ok=True)
    encoder_path = args.model.with_name("features.json")
    encoder.save(encoder_path)
    save_checkpoint(model, args.model, input_dim=encoder.dim, encoder_path=str(encoder_path), model_type="gru", sequence_length=args.seq_len)
    print(f"Saved {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
