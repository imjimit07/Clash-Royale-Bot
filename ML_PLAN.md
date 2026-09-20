# ML Upgrade Plan

This fork now uses a staged hybrid architecture. The deterministic emulator/vision
code remains in control of legality and input execution; learned policies only
propose decisions and every proposal passes a safety validator.

## Stage 1 — reliable hybrid foundation

- One card registry (`clbot/bot/card_database.py`) is the source of truth.
- Unknown card identities are never silently converted into a fake card/cost for a live deployment.
- Elixir `0` and measurement-unavailable are distinct internally.
- Deployments are confirmed before cost/history/training state is mutated.
- Recorder action events are timestamped and flushed immediately.
- A recurrent policy is reset at the start of every battle.

## Stage 2 — supervised dataset

Run:

```bash
python training/build_dataset.py <recordings-dir> data/dataset
```

The dataset includes both confirmed plays and deliberate `WAIT` decisions. By
default, learned-policy outputs are excluded so the model does not simply teach
itself its own mistakes. Use `--sources learned,...` only for deliberate iterative
training. The card target is encoded as `0=WAIT`, `1=slot0`, `2=slot1`, `3=slot2`,
`4=slot3`.
Validation splits are match-level, so decisions from one fight never leak into
both train and validation sets.

## Stage 3 — feed-forward policy

Run:

```bash
python training/train_policy.py data/dataset models/policy.pt
```

The model predicts:

1. wait or one of four hand slots;
2. one of 96 arena cells when a card is played.

Placement loss is masked for WAIT rows. Runtime inference is optional and falls
back to the deterministic tactical/legacy policies if the checkpoint is absent,
invalid, low-confidence, unaffordable, or illegal.

## Stage 4 — sequence policy

Run:

```bash
python training/train_sequence.py data/dataset models/policy_gru.pt --seq-len 8
```

The GRU sees the last eight decision states from one match. The live runtime also
supports GRU checkpoints and keeps its history isolated per battle.

## Stage 5 — reinforcement learning

`clbot/bot/ml/rl.py` now contains a generic discrete-action PPO learner. A real,
deterministic simulator is still required for meaningful Clash Royale training.
Do not use the live emulator as a high-frequency PPO environment.

The PPO action space can be flattened as:

```text
action = card_slot * 96 + placement_cell
```

with `384` possible discrete actions. A simulator adapter can later mask invalid
cards/placements and expose reward components such as tower damage, elixir
trades, and terminal win/loss.

## Live-model activation

The live bot searches for `models/policy_gru.pt` first and then
`models/policy.pt`, unless `CLBOT_ML_MODEL` is explicitly set. The feature encoder
is loaded from `CLBOT_ML_ENCODER` or `models/features.json`.

## Recommended data quality progression

Start with human demonstrations or carefully curated matches. Bot-generated
heuristic labels are useful for bootstrapping but reproduce the heuristic's
mistakes. Evaluate new checkpoints offline before enabling them for live play.
