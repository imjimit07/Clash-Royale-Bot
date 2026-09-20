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


def _evaluate(model, x_val, c_val, p_val) -> tuple[float, float, float]:
    """Val metrics: overall card acc, play-only card acc, placement acc."""
    model.eval()
    with torch.inference_mode():
        if len(x_val) == 0:
            return 0.0, 0.0, 0.0
        val_x = torch.from_numpy(x_val)
        val_c = torch.from_numpy(c_val)
        val_p = torch.from_numpy(p_val)
        card_logits, place_logits = model(val_x)
        card_acc = float((card_logits.argmax(1) == val_c).float().mean())
        play_mask = val_c != 0
        if bool(play_mask.any()):
            play_acc = float((card_logits[play_mask].argmax(1) == val_c[play_mask]).float().mean())
            place_acc = float((place_logits[play_mask].argmax(1) == val_p[play_mask]).float().mean())
        else:
            play_acc = 0.0
            place_acc = 0.0
    return card_acc, play_acc, place_acc


def _train_model(
    x_train,
    c_train,
    p_train,
    x_val,
    c_val,
    p_val,
    *,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    log_prefix: str = "",
) -> tuple[PolicyNet, float, float, float, int]:
    """Train one model; return (model at best val, card_acc, play_acc, place_acc, best_epoch).

    Best is selected on play-only accuracy (chance 0.25 over 4 slots); when
    the val split contains no plays it falls back to overall card accuracy.
    """
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(c_train), torch.from_numpy(p_train))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    counts = np.bincount(c_train, minlength=5).astype(np.float32)
    card_weights = np.ones(5, dtype=np.float32)
    nonzero = counts > 0
    card_weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    card_weights *= 5.0 / max(card_weights.sum(), 1e-6)
    model = PolicyNet(dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    card_loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(card_weights))
    placement_loss_fn = nn.CrossEntropyLoss()
    has_plays = bool((c_val != 0).any())

    best_state = None
    best_key = -1.0
    best_card = best_play = best_place = 0.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
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

        card_acc, play_acc, place_acc = _evaluate(model, x_val, c_val, p_val)
        print(
            f"{log_prefix}epoch={epoch:03d} train_loss={total/len(train_ds):.4f} "
            f"val_card_acc={card_acc:.3f} val_play_acc={play_acc:.3f} val_place_acc={place_acc:.3f}"
        )
        key = play_acc if has_plays else card_acc
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_card, best_play, best_place, best_epoch = card_acc, play_acc, place_acc, epoch

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"{log_prefix}best val_play_acc={best_play:.3f} at epoch {best_epoch} — checkpointing best, not last")
    return model, best_card, best_play, best_place, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path, default=Path("models/policy.pt"), nargs="?")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logo",
        action="store_true",
        help="Leave-one-game-out evaluation: train one model per held-out game, "
        "report mean±std val_play_acc, save nothing.",
    )
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
    game_ids = np.asarray([str(g) for g in games])

    if args.logo:
        unique_games = sorted(set(game_ids.tolist()))
        if len(unique_games) < 2:
            raise SystemExit("LOGO needs at least 2 games.")
        fold_accs: list[float] = []
        for fold, held_out in enumerate(unique_games):
            val_mask = game_ids == held_out
            train_mask = ~val_mask
            torch.manual_seed(args.seed + fold)
            _, _, play_acc, _, _ = _train_model(
                features[train_mask],
                card_targets[train_mask],
                placement_targets[train_mask],
                features[val_mask],
                card_targets[val_mask],
                placement_targets[val_mask],
                dim=encoder.dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                log_prefix=f"[logo {held_out}] ",
            )
            fold_accs.append(play_acc)
        mean, std = float(np.mean(fold_accs)), float(np.std(fold_accs))
        print(f"LOGO val_play_acc = {mean:.3f} ± {std:.3f} over {len(fold_accs)} folds")
        if std > mean:
            print("Std exceeds mean — not enough data to conclude anything. Record more matches.")
        return 0

    train_idx, val_idx = split_indices_by_game(games)
    x_train, x_val = features[train_idx], features[val_idx]
    c_train, c_val = card_targets[train_idx], card_targets[val_idx]
    p_train, p_val = placement_targets[train_idx], placement_targets[val_idx]
    print(f"train_rows={len(train_idx)} val_rows={len(val_idx)} train_games={len(set(str(games[i]) for i in train_idx))} val_games={len(set(str(games[i]) for i in val_idx))}")

    model, _, best_play, _, _ = _train_model(
        x_train,
        c_train,
        p_train,
        x_val,
        c_val,
        p_val,
        dim=encoder.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

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
    print(f"Saved {args.model} (best val_play_acc={best_play:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
