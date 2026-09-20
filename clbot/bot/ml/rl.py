"""Environment-neutral PPO utilities for Stage 5 reinforcement learning.

The live emulator is intentionally not an RL environment. A deterministic
simulator must implement ``reset`` and ``step`` before PPO can improve the
policy. The action space is a flattened ``card_slot * 96 + placement_cell``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from clbot.bot.ml.policy import ActorCriticNet, require_torch


class ClashEnv(Protocol):
    observation_dim: int
    action_dim: int

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]: ...


@dataclass(slots=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 256
    rollout_steps: int = 2048
    max_grad_norm: float = 0.5


def validate_env(env: ClashEnv) -> None:
    if env.observation_dim <= 0:
        raise ValueError("Environment observation_dim must be positive")
    if env.action_dim <= 1:
        raise ValueError("Environment action_dim must be > 1")


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and bootstrapped returns."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if index == len(rewards) - 1:
            next_v = float(next_value)
        else:
            next_v = float(values[index + 1])
        non_terminal = 1.0 - float(dones[index])
        delta = float(rewards[index]) + gamma * next_v * non_terminal - float(values[index])
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[index] = gae
    returns = advantages + values.astype(np.float32)
    return advantages, returns


def flatten_action(card_index: int, placement_class: int, placement_classes: int = 96) -> int:
    card_index = int(card_index)
    placement_class = int(placement_class)
    if not 0 <= card_index < 4:
        raise ValueError("card_index must be in [0, 3]")
    if not 0 <= placement_class < placement_classes:
        raise ValueError(f"placement_class must be in [0, {placement_classes - 1}]")
    return card_index * placement_classes + placement_class


def unflatten_action(action: int, placement_classes: int = 96) -> tuple[int, int]:
    action = int(action)
    if not 0 <= action < 4 * placement_classes:
        raise ValueError("flattened action is outside the policy action space")
    return action // placement_classes, action % placement_classes


def _require_torch_components():
    torch = require_torch()
    return torch


class PPOAgent:
    """Small, reusable PPO implementation for a discrete-action simulator."""

    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig | None = None, device: str = "cpu") -> None:
        torch = _require_torch_components()
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.model = ActorCriticNet(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        torch = _require_torch_components()
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            logits, value = self.model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, observations, actions, old_log_probs, advantages, returns) -> dict[str, float]:
        torch = _require_torch_components()
        self.model.train()
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_lp = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = len(obs)
        indices = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0.0}
        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.config.batch_size):
                batch = indices[start : start + self.config.batch_size]
                logits, values = self.model(obs[batch])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t[batch])
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - old_lp[batch])
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = -torch.min(ratio * adv[batch], clipped * adv[batch]).mean()
                value_loss = 0.5 * (ret[batch] - values).pow(2).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["updates"] += 1.0
        if metrics["updates"]:
            for key in ("policy_loss", "value_loss", "entropy"):
                metrics[key] /= metrics["updates"]
        return metrics

    def collect_rollout(self, env: ClashEnv, steps: int | None = None) -> tuple[dict[str, np.ndarray], float]:
        validate_env(env)
        steps = int(steps or self.config.rollout_steps)
        obs = np.asarray(env.reset(), dtype=np.float32)
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        log_probs: list[float] = []
        episode_reward = 0.0

        for _ in range(steps):
            action, log_prob, value = self.act(obs)
            next_obs, reward, done, _info = env.step(action)
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(value)
            log_probs.append(log_prob)
            episode_reward += float(reward)
            obs = np.asarray(env.reset() if done else next_obs, dtype=np.float32)

        _, _, next_value = self.act(obs, deterministic=True)
        advantages, returns = compute_gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            next_value,
            self.config.gamma,
            self.config.gae_lambda,
        )
        batch = {
            "observations": np.stack(observations).astype(np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "old_log_probs": np.asarray(log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }
        return batch, episode_reward


def train_ppo(env: ClashEnv, updates: int = 10, config: PPOConfig | None = None, device: str = "cpu") -> PPOAgent:
    """Train a generic discrete PPO agent on a simulator environment."""
    validate_env(env)
    agent = PPOAgent(env.observation_dim, env.action_dim, config=config, device=device)
    for update_idx in range(1, int(updates) + 1):
        batch, episode_reward = agent.collect_rollout(env)
        metrics = agent.update(**batch)
        print(
            f"update={update_idx:03d} reward={episode_reward:.3f} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f}"
        )
    return agent


def explain_stage5() -> str:
    return (
        "Stage 5 provides a complete generic PPO learner. Connect a deterministic "
        "Clash Royale simulator; do not train PPO by repeatedly driving the live emulator."
    )
