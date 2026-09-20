"""PyTorch policy networks for supervised and recurrent training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # Optional runtime dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# Card head: 0 = WAIT, 1..4 = visible hand slot 0..3.
CARD_ACTIONS = 5
DEFAULT_GRID_COLS = 12
DEFAULT_GRID_ROWS = 8
PLACEMENT_CLASSES = DEFAULT_GRID_COLS * DEFAULT_GRID_ROWS


def require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for ML training/inference. Install `clbot[ml]`.")
    return torch


if nn is not None:

    class PolicyNet(nn.Module):
        """Feed-forward multi-task policy: card slot + placement cell."""

        def __init__(self, input_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 128),
                nn.ReLU(),
            )
            self.card_head = nn.Linear(128, CARD_ACTIONS)
            self.place_head = nn.Linear(128, PLACEMENT_CLASSES)

        def forward(self, x):
            z = self.backbone(x)
            return self.card_head(z), self.place_head(z)


    class SequencePolicyNet(nn.Module):
        """GRU policy over a short decision history."""

        def __init__(self, input_dim: int, hidden: int = 192, layers: int = 2) -> None:
            super().__init__()
            self.input = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
            self.gru = nn.GRU(
                128,
                hidden,
                num_layers=layers,
                batch_first=True,
                dropout=0.1 if layers > 1 else 0.0,
            )
            self.card_head = nn.Linear(hidden, CARD_ACTIONS)
            self.place_head = nn.Linear(hidden, PLACEMENT_CLASSES)

        def forward(self, x, lengths=None):
            z = self.input(x)
            output, _ = self.gru(z)
            last = output[:, -1, :]
            return self.card_head(last), self.place_head(last)


    class ActorCriticNet(nn.Module):
        """Generic discrete-action actor/critic used by Stage 5 PPO."""

        def __init__(self, input_dim: int, action_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 128),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(128, action_dim)
            self.value_head = nn.Linear(128, 1)

        def forward(self, x):
            z = self.backbone(x)
            return self.policy_head(z), self.value_head(z).squeeze(-1)


else:

    class PolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


    class SequencePolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


    class ActorCriticNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


def save_checkpoint(
    model,
    path: str | Path,
    *,
    input_dim: int,
    encoder_path: str,
    model_type: str,
    sequence_length: int | None = None,
) -> None:
    require_torch()
    payload = {
        "schema": 2,
        "model_type": model_type,
        "input_dim": int(input_dim),
        "encoder_path": encoder_path,
        "sequence_length": None if sequence_length is None else int(sequence_length),
        "card_actions": CARD_ACTIONS,
        "placement_classes": PLACEMENT_CLASSES,
        "state_dict": model.state_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, device: str = "cpu"):
    require_torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    input_dim = int(payload["input_dim"])
    model_type = str(payload.get("model_type", "feedforward"))
    checkpoint_schema = int(payload.get("schema", 1))
    if checkpoint_schema >= 2 and int(payload.get("card_actions", CARD_ACTIONS)) != CARD_ACTIONS and model_type != "actor_critic":
        raise ValueError("Checkpoint card action space does not match this runtime")
    if model_type == "gru":
        model = SequencePolicyNet(input_dim)
    elif model_type == "actor_critic":
        action_dim = int(payload["action_dim"])
        model = ActorCriticNet(input_dim, action_dim)
    else:
        model = PolicyNet(input_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _numpy_logits(model, x):
    with torch.inference_mode():
        card_logits, place_logits = model(x)
        return card_logits.squeeze(0).cpu().numpy(), place_logits.squeeze(0).cpu().numpy()


def predict_logits(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer one state with a feed-forward policy."""
    require_torch()
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        return _numpy_logits(model, x)


def predict_sequence_logits(model, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer a [time, feature] sequence with a recurrent policy."""
    require_torch()
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, features], got {values.shape}")
    with torch.inference_mode():
        x = torch.from_numpy(values).unsqueeze(0)
        return _numpy_logits(model, x)
