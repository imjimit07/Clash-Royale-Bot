# ML implementation notes

## Implemented stages

### 1. Reliable hybrid foundation

- Canonical card lookup via `lookup_card()`.
- Live deployment refuses unknown cards instead of charging/pretending a fallback cost.
- Unified elixir path uses `ElixirTracker` plus optional scanner results.
- Successful deployment is verified before updating elixir/history/training logs.
- Battle-local ML/recent-card state is reset per match.
- Purple-card colour comparison uses signed integer arithmetic instead of uint8 wrap-around.
- Hold decisions are throttled so repeated idle frames do not flood the dataset.

### 2. Supervised dataset

- Recorder now writes structured `play` and `hold` decision events.
- Events are flushed immediately, reducing data loss on crashes.
- Dataset schema is `3` and supports `0=WAIT`, `1..4=hand slot` labels.
- Match-level train/validation splitting prevents decision leakage across matches.
- Dataset building can filter by policy source; `learned` is excluded by default to reduce self-training feedback loops.

### 3. Feed-forward policy

- `PolicyNet` is a real PyTorch multi-head neural network.
- Card head predicts WAIT or one of four hand slots.
- Placement head predicts a 12x8 arena grid (96 classes).
- Card loss uses inverse-square-root class balancing.
- Placement loss is applied only to actual play rows, not WAIT rows.
- Checkpoints save model type, feature dimension and action-space metadata.
- Live inference is optional and falls back safely when no usable checkpoint exists.

### 4. Sequence policy

- `SequencePolicyNet` adds a GRU over eight decision states by default.
- Windows are built only inside a single match.
- Train/validation matches remain disjoint.
- The live runtime supports GRU checkpoints and resets its history between battles.
- Checkpoints store the trained sequence length.

### 5. Reinforcement learning

- Added a reusable discrete PPO implementation in `clbot/bot/ml/rl.py`.
- Added GAE, rollout collection, clipped policy loss, value loss and entropy regularisation.
- Action flattening maps `card_slot * 96 + placement_cell` into a 384-action simulator space.
- `training/train_ppo.py` connects the trainer to any deterministic simulator exposing `make_env()`.
- A Clash Royale simulator is not included, so PPO is not trained from the live emulator.

## Validation performed

- Full source tree passed Python bytecode compilation with `python -m compileall`.
- Feature-encoder dimension smoke check: `869` features.
- Action-validator smoke checks passed for a valid play and an unaffordable play.
- Recorder smoke check passed for both play and hold events.
- `pytest` could not complete in the supplied container because the test process was OS-killed (status 9); this appears to be an environment resource limitation rather than a Python syntax failure.
- `uv lock --offline` could not regenerate `uv.lock` because the container lacks the required managed Python 3.12 interpreter. Run `uv lock` on the development machine after pulling the updated `pyproject.toml`.

## First run after extracting the project

Install the ML extra:

```bash
uv lock
uv sync --extra ml
```

Then record completed matches with the updated bot and build the dataset:

```bash
python training/build_dataset.py <recordings-dir> data/dataset
```

Train the feed-forward model first:

```bash
python training/train_policy.py data/dataset models/policy.pt
```

Then train the GRU model:

```bash
python training/train_sequence.py data/dataset models/policy_gru.pt --seq-len 8
```

The runtime prefers `models/policy_gru.pt` when present. Remove the checkpoint or set `CLBOT_ML_MODEL` to change what the live bot loads.
