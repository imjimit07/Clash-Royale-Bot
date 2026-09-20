# Model directory

Trained model checkpoints are intentionally not committed/generated with this source archive.

After collecting recordings, build a dataset and train:

```bash
python training/build_dataset.py <recordings-dir> data/dataset
python training/train_policy.py data/dataset models/policy.pt
python training/train_sequence.py data/dataset models/policy_gru.pt --seq-len 8
```

The live bot prefers `policy_gru.pt` when present, then `policy.pt`.
Set `CLBOT_ML_MODEL` and `CLBOT_ML_ENCODER` for explicit paths.
