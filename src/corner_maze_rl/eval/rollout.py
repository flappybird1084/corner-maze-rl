"""DT rollout: take a trained model + an env + a target return, and run.

Implements the inference loop in plan §4.4. The student-set ``target_return``
is decremented by each observed reward, so the RTG token at step *t* is
``target_return - sum(rewards[:t])``.

Maintains a sliding context of K recent ``(rtg, state, action)`` triples,
front-padded with zeros while the trajectory is still warming up.
"""
from __future__ import annotations

import io
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from corner_maze_rl.models.decision_transformer import DecisionTransformer


@dataclass
class RolloutResult:
    actions: list[int]
    rewards: list[float]
    rtg_seq: list[float]
    positions: list[tuple[int, int, int]]
    terminated: bool
    truncated: bool

    @property
    def total_return(self) -> float:
        return float(sum(self.rewards))


def _action_one_hot(action: int, num_actions: int) -> np.ndarray:
    v = np.zeros(num_actions, dtype=np.float32)
    if 0 <= action < num_actions:
        v[action] = 1.0
    return v


def _agent_pose(env) -> tuple[int, int, int]:
    """Best-effort: pull (x, y, dir) from the unwrapped env."""
    raw = env
    while hasattr(raw, "env") and not hasattr(raw, "agent_pos"):
        raw = raw.env
    pos = getattr(raw, "agent_pos", (0, 0))
    direction = int(getattr(raw, "agent_dir", 0))
    return int(pos[0]), int(pos[1]), direction


def rollout_dt(
    model: DecisionTransformer,
    env,
    encoder,
    *,
    target_return: float,
    max_steps: int = 6000,
    deterministic: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
    use_action_mask: bool = True,
    suppress_env_stdout: bool = True,
    device: str | torch.device = "cpu",
) -> RolloutResult:
    """Roll out a Decision Transformer in *env*.

    Parameters
    ----------
    model : DecisionTransformer
        Trained DT. ``model.cfg.context_size`` and ``cfg.num_actions`` drive
        the rollout shapes.
    env
        Wrapped env exposing ``reset``, ``step``, and (optionally)
        ``get_action_mask``.
    encoder
        StateEncoder; ``encode(x, y, direction)`` returns a vector of
        length ``model.cfg.embed_dim``.
    target_return
        Initial RTG. Decremented by each observed reward each step.
    deterministic
        If True, picks ``argmax(logits)``; else samples with ``temperature``.
    use_action_mask
        If True and the env exposes ``get_action_mask``, illegal actions
        are masked to -inf before argmax/sampling.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    cfg = model.cfg
    K = cfg.context_size
    A = cfg.num_actions
    D = cfg.embed_dim

    # Sliding context buffers, padded with zeros up front.
    rtg_buf = deque([np.zeros(1, dtype=np.float32)] * K, maxlen=K)
    state_buf = deque([np.zeros(D, dtype=np.float32)] * K, maxlen=K)
    action_buf = deque([np.zeros(A, dtype=np.float32)] * K, maxlen=K)
    pos_buf = deque([np.zeros(D, dtype=np.float32)] * K, maxlen=K)

    actions_taken: list[int] = []
    rewards_seen: list[float] = []
    rtg_seq: list[float] = []
    positions: list[tuple[int, int, int]] = []

    old_stdout = sys.stdout
    if suppress_env_stdout:
        sys.stdout = io.StringIO()

    try:
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        running_rtg = float(target_return)
        terminated = False
        truncated = False

        for _ in range(max_steps):
            x, y, d = _agent_pose(env)
            positions.append((x, y, d))
            s_vec = encoder.encode(x, y, d).astype(np.float32, copy=False)
            state_buf.append(s_vec)
            pos_buf.append(s_vec)
            rtg_buf.append(np.array([running_rtg], dtype=np.float32))
            rtg_seq.append(running_rtg)

            rtg_t = torch.from_numpy(np.stack(list(rtg_buf))).unsqueeze(0).to(device)
            state_t = torch.from_numpy(np.stack(list(state_buf))).unsqueeze(0).to(device)
            action_t = torch.from_numpy(np.stack(list(action_buf))).unsqueeze(0).to(device)
            pos_t = torch.from_numpy(np.stack(list(pos_buf))).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(rtg_t, state_t, action_t, pos_vecs=pos_t)
            # Predict from the LAST token in the context window.
            last_logits = logits[0, -1].cpu().numpy().astype(np.float64)

            if use_action_mask and hasattr(env, "get_action_mask"):
                mask = np.asarray(env.get_action_mask(), dtype=bool)
                # Pad / truncate mask to A
                mask = mask[:A] if len(mask) >= A else np.concatenate([mask, [True] * (A - len(mask))])
                last_logits = np.where(mask, last_logits, -np.inf)

            if deterministic:
                action = int(np.argmax(last_logits))
            else:
                z = last_logits / max(temperature, 1e-6)
                z = z - np.max(z)
                p = np.exp(z)
                p = p / p.sum()
                action = int(rng.choice(A, p=p))

            action_buf.append(_action_one_hot(action, A))
            _, r, term, trunc, _ = env.step(action)
            actions_taken.append(action)
            rewards_seen.append(float(r))
            running_rtg -= float(r)

            if term or trunc:
                terminated = bool(term)
                truncated = bool(trunc)
                break
    finally:
        if suppress_env_stdout:
            sys.stdout = old_stdout

    return RolloutResult(
        actions=actions_taken,
        rewards=rewards_seen,
        rtg_seq=rtg_seq,
        positions=positions,
        terminated=terminated,
        truncated=truncated,
    )
