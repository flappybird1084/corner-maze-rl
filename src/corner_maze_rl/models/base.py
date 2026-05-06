"""TrainableAgent Protocol per plan §6.1.

A new model only needs to implement this interface to plug into
``train.runner``. The env never changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TrainableAgent(ABC):
    """Per plan §6.1. Used by PPO and SR; DT plays a separate role
    (offline supervised training on yoked data; see eval/rollout for
    online inference)."""

    @abstractmethod
    def select_action(
        self, state: np.ndarray, action_mask: list[bool]
    ) -> tuple[int, dict[str, Any]]:
        """Return (action, info_dict). ``info`` may be empty for simple agents."""

    @abstractmethod
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        **kwargs,
    ) -> None:
        """Store a transition in the agent's buffer."""

    @abstractmethod
    def update(
        self, next_state: np.ndarray, next_done: bool
    ) -> dict[str, float]:
        """Update the policy. Return a dict of loss metrics."""

    def is_ready_to_update(self) -> bool:
        """True when enough experience has been collected to call ``update``."""
        return False

    def save(self, path: str) -> None:
        """Save agent state to disk."""

    def load(self, path: str) -> None:
        """Load agent state from disk."""
