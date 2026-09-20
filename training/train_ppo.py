#!/usr/bin/env python3
"""Train the Stage-5 PPO learner against a simulator module.

The module must expose ``make_env()`` returning an object with:
``observation_dim``, ``action_dim``, ``reset()`` and ``step(action)``.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from clbot.bot.ml.policy import require_torch
from clbot.bot.ml.rl import PPOConfig, explain_stage5, train_ppo, validate_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_module", help="Python module exposing make_env()")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("models") / "ppo_policy.pt")
    args = parser.parse_args()

    module = importlib.import_module(args.env_module)
    make_env = getattr(module, "make_env", None)
    if make_env is None:
        raise SystemExit("Environment module must expose make_env().")
    env = make_env()
    validate_env(env)
    print(explain_stage5())
    print(f"Environment accepted: observation_dim={env.observation_dim} action_dim={env.action_dim}")

    config = PPOConfig(
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    agent = train_ppo(env, updates=args.updates, config=config)
    torch = require_torch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "model_type": "actor_critic",
            "input_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "state_dict": agent.model.state_dict(),
        },
        args.output,
    )
    print(f"Saved PPO checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
