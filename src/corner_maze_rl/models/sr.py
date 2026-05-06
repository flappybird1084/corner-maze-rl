"""Successor Representation agent (linear feature-based, de Cothi 2022).

Ported from legacy ``custom_rl.py::SRAgent``. Decomposes value into a
predictive map M (per-action expected future state occupancy) and reward
weights w:  ``Q(s,a) = phi(s) @ M[a] @ w``.

This agent reliably *fails* to learn the corner-maze under tabular state
features — see ``md/sr-yoked-negative-results.md`` and plan §6.4. Shipped
as an instructive negative-results case for student comparisons (plan §10.3).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from corner_maze_rl.encoders.state_vectors import (
    NUM_LOCATIONS,
    build_position_projection_matrix,
)
from corner_maze_rl.models.base import TrainableAgent


class SRAgent(TrainableAgent):
    """Linear feature-based SR with TD(0) updates and ε-greedy action.

    Adapted for continuous multi-trial episodes — *no* trial-boundary
    resets. Uses a Q-learning-style off-policy SR update (greedy next
    action for the M target).

    Two reward-weight modes:
      * Default (full): ``w`` lives in obs-space (236-D for n_wm=10).
      * ``position_only_w=True``: ``w`` lives in 89-D position-only space
        via a fixed projection. M still operates in full obs-space; the
        successor prediction is projected to position-only before dotting
        with ``w``.
    """

    def __init__(
        self,
        obs_dim: int = 236,
        action_dim: int = 5,
        gamma: float = 0.99,
        lr_m: float = 0.01,
        lr_w: float = 0.05,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        position_only_w: bool = False,
        n_wm_units: int = 10,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_m = lr_m
        self.lr_w = lr_w
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.position_only_w = position_only_w

        # M[a] initialised to identity → "successor of state s is s itself".
        self.M = np.stack([np.eye(obs_dim, dtype=np.float32) for _ in range(action_dim)])

        if position_only_w:
            self._pos_proj = build_position_projection_matrix(n_wm_units)
            self._w_dim = NUM_LOCATIONS + 4 * n_wm_units
            self.w = np.zeros(self._w_dim, dtype=np.float32)
        else:
            self._pos_proj = None
            self._w_dim = obs_dim
            self.w = np.zeros(obs_dim, dtype=np.float32)

        # Single-step online TD buffer.
        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_reward: float | None = None
        self._last_done: bool | None = None
        self._has_experience = False

    # ------------------------------------------------------------------
    # Q-values
    # ------------------------------------------------------------------

    def _project_to_position(self, state: np.ndarray) -> np.ndarray:
        return state @ self._pos_proj

    def _compute_q_values(self, state: np.ndarray) -> np.ndarray:
        q = np.empty(self.action_dim, dtype=np.float32)
        if self.position_only_w:
            for a in range(self.action_dim):
                successor = state @ self.M[a]
                q[a] = (successor @ self._pos_proj) @ self.w
        else:
            for a in range(self.action_dim):
                q[a] = state @ self.M[a] @ self.w
        return q

    # ------------------------------------------------------------------
    # TrainableAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray, action_mask: list[bool]
    ) -> tuple[int, dict[str, Any]]:
        q = self._compute_q_values(state)
        masked = q.copy()
        valid = []
        for i, allowed in enumerate(action_mask):
            if not allowed:
                masked[i] = float("-inf")
            else:
                valid.append(i)
        if not valid:
            valid = [0]

        if np.random.random() < self.epsilon:
            action = valid[np.random.randint(len(valid))]
        else:
            action = int(np.argmax(masked))
        return action, {"q_values": q}

    def add_experience(self, state, action, reward, done, **kwargs):
        self._last_state = state
        self._last_action = action
        self._last_reward = reward
        self._last_done = done
        self._has_experience = True

    def is_ready_to_update(self) -> bool:
        return self._has_experience

    def update(self, next_state: np.ndarray, next_done: bool) -> dict[str, float]:
        phi_s = self._last_state
        a = self._last_action
        r = self._last_reward
        assert phi_s is not None and a is not None and r is not None

        # --- w update (reward-weight regression) ---
        if self.position_only_w:
            phi_pos = self._project_to_position(phi_s)
            w_error = r - (phi_pos @ self.w)
            self.w += self.lr_w * w_error * phi_pos
        else:
            w_error = r - (phi_s @ self.w)
            self.w += self.lr_w * w_error * phi_s
        np.clip(self.w, -10.0, 10.0, out=self.w)

        # --- M update (Q-learning-style off-policy target) ---
        if self._last_done:
            sr_target = phi_s
        else:
            q_next = self._compute_q_values(next_state)
            a_prime = int(np.argmax(q_next))
            sr_target = phi_s + self.gamma * (self.M[a_prime] @ next_state)

        sr_pred = self.M[a] @ phi_s
        sr_error = sr_target - sr_pred
        self.M[a] += self.lr_m * np.outer(phi_s, sr_error)
        np.clip(self.M[a], -10.0, 10.0, out=self.M[a])

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._has_experience = False

        return {
            "w_error": float(abs(w_error)),
            "sr_error": float(np.mean(np.abs(sr_error))),
            "epsilon": float(self.epsilon),
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(
            path,
            M=self.M,
            w=self.w,
            epsilon=np.array(self.epsilon),
            position_only_w=np.array(self.position_only_w),
        )

    def load(self, path: str) -> None:
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        self.M = data["M"]
        self.w = data["w"]
        self.epsilon = float(data["epsilon"])
