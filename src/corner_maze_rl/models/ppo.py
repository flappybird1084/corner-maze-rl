"""PPO with GAE and action masking, ported from legacy custom_rl.py.

Adapted for the new package layout. Generic over ``obs_dim`` so it works
with any encoder configuration (236-D tabular, 60-D grid-cells, etc.).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from corner_maze_rl.models.base import TrainableAgent


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Flexible actor-critic with optional shared hidden layer.

    Initialization follows plan-feedback memory: orthogonal with
    gain=sqrt(2) for ReLU, 0.01 for actor head (near-uniform initial
    policy), 1.0 for critic head.
    """

    def __init__(self, obs_dim: int, action_dim: int, n_hidden_units: int = 0):
        super().__init__()
        if n_hidden_units > 0:
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, n_hidden_units), nn.ReLU()
            )
            self.actor = nn.Linear(n_hidden_units, action_dim)
            self.critic = nn.Linear(n_hidden_units, 1)
        else:
            self.shared = nn.Identity()
            self.actor = nn.Linear(obs_dim, action_dim)
            self.critic = nn.Linear(obs_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPOAgent(TrainableAgent):
    """PPO with GAE-λ and action masking.

    Action mask is applied as ``logits[masked] = -inf`` before sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_hidden_units: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        n_steps: int = 256,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy = ActorCritic(obs_dim, action_dim, n_hidden_units)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self._obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self._actions = np.zeros(n_steps, dtype=np.int64)
        self._log_probs = np.zeros(n_steps, dtype=np.float32)
        self._rewards = np.zeros(n_steps, dtype=np.float32)
        self._dones = np.zeros(n_steps, dtype=np.float32)
        self._values = np.zeros(n_steps, dtype=np.float32)
        self._step = 0

    def select_action(
        self, state: np.ndarray, action_mask: list[bool]
    ) -> tuple[int, dict[str, Any]]:
        with torch.no_grad():
            logits, value = self.policy(torch.as_tensor(state, dtype=torch.float32))
            masked = logits.clone()
            for i, allowed in enumerate(action_mask):
                if not allowed:
                    masked[i] = float("-inf")
            dist = Categorical(logits=masked)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), {"value": float(value.item()), "log_prob": float(log_prob.item())}

    def add_experience(self, state, action, reward, done, **kwargs):
        if self._step >= self.n_steps:
            return  # silently drop after buffer full; caller should call update
        self._obs[self._step] = state
        self._actions[self._step] = action
        self._rewards[self._step] = reward
        self._dones[self._step] = float(done)
        self._values[self._step] = kwargs.get("value", 0.0)
        self._log_probs[self._step] = kwargs.get("log_prob", 0.0)
        self._step += 1

    def is_ready_to_update(self) -> bool:
        return self._step >= self.n_steps

    def update(self, next_state: np.ndarray, next_done: bool) -> dict[str, float]:
        with torch.no_grad():
            next_value = self.policy(
                torch.as_tensor(next_state, dtype=torch.float32)
            )[1].item()

            advantages = np.zeros_like(self._rewards)
            last_gae = 0.0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_term = 1.0 - float(next_done)
                    next_val = next_value
                else:
                    next_non_term = 1.0 - self._dones[t + 1]
                    next_val = self._values[t + 1]
                delta = self._rewards[t] + self.gamma * next_val * next_non_term - self._values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_term * last_gae
            returns = advantages + self._values

        b_obs = torch.as_tensor(self._obs, dtype=torch.float32)
        b_actions = torch.as_tensor(self._actions, dtype=torch.long)
        b_log_probs = torch.as_tensor(self._log_probs, dtype=torch.float32)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32)
        b_returns = torch.as_tensor(returns, dtype=torch.float32)
        inds = np.arange(self.n_steps)

        n_updates = 0
        sum_pg = sum_v = sum_ent = 0.0

        for _ in range(self.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.n_steps, self.batch_size):
                mb = inds[start: start + self.batch_size]
                logits, values = self.policy(b_obs[mb])
                dist = Categorical(logits=logits)
                log_ratio = dist.log_prob(b_actions[mb]) - b_log_probs[mb]
                ratio = torch.exp(log_ratio)
                adv = b_advantages[mb]

                pg = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef),
                ).mean()
                v_loss = 0.5 * ((values.view(-1) - b_returns[mb]) ** 2).mean()
                ent = dist.entropy().mean()

                loss = pg - self.ent_coef * ent + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                sum_pg += pg.item()
                sum_v += v_loss.item()
                sum_ent += ent.item()
                n_updates += 1

        self._step = 0  # reset rollout buffer

        n_updates = max(n_updates, 1)
        return {
            "pg_loss": sum_pg / n_updates,
            "v_loss": sum_v / n_updates,
            "entropy": sum_ent / n_updates,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
