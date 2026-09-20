#!/usr/bin/env python3
"""Train the first real feed-forward policy from recorded decisions."""
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
from clbot.bot.ml.policy import PolicyNet, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path, default=Path("models/policy.pt"), nargs="?")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features = np.load(args.dataset / "features.npy")
    card_targets = np.load(args.dataset / "card_targets.npy")
    placement_targets = np.load(args.dataset / "placement_targets.npy")
    encoder = FeatureEncoder.load(args.dataset / "features.json")
    if len(features) < 8:
        raise SystemExit("Need at least 8 samples before training a policy.")

    games = json.loads((args.dataset / "games.json").read_text(encoding="utf-8"))
    if len(games) != len(features):
        raise SystemExit("Dataset games.json length does not match features.npy")
    train_idx, val_idx = split_indices_by_game(games)
    x_train, x_val = features[train_idx], features[val_idx]
    c_train, c_val = card_targets[train_idx], card_targets[val_idx]
    p_train, p_val = placement_targets[train_idx], placement_targets[val_idx]
    print(f"train_rows={len(train_idx)} val_rows={len(val_idx)} train_games={len(set(str(games[i]) for i in train_idx))} val_games={len(set(str(games[i]) for i in val_idx))}")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(c_train), torch.from_numpy(p_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    counts = np.bincount(c_train, minlength=5).astype(np.float32)
    card_weights = np.ones(5, dtype=np.float32)
    nonzero = counts > 0
    card_weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    card_weights *= 5.0 / max(card_weights.sum(), 1e-6)
    model = PolicyNet(encoder.dim)
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

        model.eval()
        with torch.inference_mode():
            if len(x_val):
                val_x = torch.from_numpy(x_val)
                val_c = torch.from_numpy(c_val)
                val_p = torch.from_numpy(p_val)
                card_logits, place_logits = model(val_x)
                card_acc = float((card_logits.argmax(1) == val_c).float().mean())
                play_mask = val_c != 0
                place_acc = float((place_logits[play_mask].argmax(1) == val_p[play_mask]).float().mean()) if bool(play_mask.any()) else 0.0
            else:
                card_acc = place_acc = 0.0
        print(f"epoch={epoch:03d} train_loss={total/len(train_ds):.4f} card_acc={card_acc:.3f} placement_acc={place_acc:.3f}")

    args.model.parent.mkdir(parents=True, exist_ok=True)
    encoder_path = args.model.with_name("features.json")
    encoder.save(encoder_path)
    save_checkpoint(
        model,
        args.model,
        input_dim=encoder.dim,
        encoder_path=str(encoder_path),
        model_type="feedforward",
    )
    print(f"Saved {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
