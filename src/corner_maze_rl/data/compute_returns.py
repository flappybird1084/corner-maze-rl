"""Compute per-step reward and per-trial-with-ITI-start return-to-go.

This module replays a yoked action sequence through ``CornerMazeEnv`` and
emits a per-step DataFrame with ``reward`` and ``return_to_go`` columns
following plan §4.

RTG window definition (per-trial, ITI-start):
  * Trial *T*'s window covers ``[first step of ITI preceding T, last step of T]``.
  * The ITI before a trial belongs to that upcoming trial's RTG.
  * Operationally: we capture ``env.trial_count`` *before* each step. That
    counter advances when a trial ends (correct well visit). The ITI that
    follows is therefore observed under the new (next-trial) count, so any
    step in that ITI naturally rolls up into the next trial's RTG.

The output adds these columns to the input row schema:

  * ``reward``       — float32, env-derived per-step reward.
  * ``trial_idx``    — int, ``env.trial_count`` *before* the step.
  * ``state_type``   — int, the env phase tag (STATE_BASE..STATE_ITI).
  * ``return_to_go`` — float32, suffix sum of ``reward`` within the same
                       ``trial_idx`` window.

The session→env mapping (training_group × yoked_session_type → env paradigm
and goal location) is in :mod:`corner_maze_rl.data.session_types`; this
module just consumes a callable that returns an env.
"""
from __future__ import annotations

import io
import sys
from typing import Callable

import numpy as np
import pandas as pd


REWARD_COL = "reward"
RTG_COL = "return_to_go"
TRIAL_IDX_COL = "trial_idx"
STATE_TYPE_COL = "state_type"


def _read_state_type(env) -> int:
    """Best-effort capture of the current env phase tag.

    grid_configuration_sequence entries are layout tuples for pretrial/trial
    (state_type at index 0), and a 3-tuple of (layoutA, layoutB, layoutC) for
    ITI. For ITI we resolve to the active sub-config via _iti_config_idx.

    Returns -1 if the structure can't be parsed (defensive — state_type is
    annotation, not load-bearing for RTG).
    """
    try:
        entry = env.grid_configuration_sequence[env.sequence_count]
        head = entry[0]
        if isinstance(head, (int, float)):
            return int(head)
        # ITI: head is itself a layout tuple. Pick the active sub-config.
        idx = int(getattr(env, "_iti_config_idx", 0) or 0)
        sub = entry[idx] if idx < len(entry) else entry[0]
        return int(sub[0])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Pure RTG arithmetic
# ---------------------------------------------------------------------------

def per_cycle_suffix_sum(
    rewards: np.ndarray,
    cycle_ids: np.ndarray,
) -> np.ndarray:
    """Return-to-go: backward suffix sum of ``rewards``, resetting at each
    change of ``cycle_ids``.

    Operationally: ``rtg[t] = rewards[t] + (rtg[t+1] if cycle_ids[t]==cycle_ids[t+1] else 0)``.
    Pure numpy; no env required.
    """
    n = len(rewards)
    if n != len(cycle_ids):
        raise ValueError("rewards and cycle_ids must have equal length")
    rtg = np.zeros(n, dtype=np.float32)
    if n == 0:
        return rtg
    rtg[-1] = rewards[-1]
    for t in range(n - 2, -1, -1):
        if cycle_ids[t] == cycle_ids[t + 1]:
            rtg[t] = rewards[t] + rtg[t + 1]
        else:
            rtg[t] = rewards[t]
    return rtg


# ---------------------------------------------------------------------------
# Env replay
# ---------------------------------------------------------------------------

def compute_returns_for_session(
    actions: pd.DataFrame,
    env_factory: Callable[[], "CornerMazeEnv"],  # noqa: F821
    *,
    action_col: str = "action",
    seed: int | None = None,
    suppress_env_stdout: bool = True,
) -> pd.DataFrame:
    """Replay a yoked session through ``env_factory()``'s env.

    Parameters
    ----------
    actions
        Per-step yoked DataFrame for one session. Must contain ``action_col``.
        Order is preserved.
    env_factory
        Zero-arg callable that returns a fresh ``CornerMazeEnv`` configured
        for this session (correct ``session_type``, orientation, goal). Caller
        is responsible for the (subject, session) → env-kwargs mapping.
    action_col
        Column name for the action ints.
    seed
        Passed to ``env.reset(seed=...)`` for determinism. ``None`` lets the
        env pick its own.
    suppress_env_stdout
        The legacy env prints agent_pos every step; suppress it so a 1M-row
        replay doesn't flood stdout. True by default.

    Returns
    -------
    pd.DataFrame
        Copy of *actions* (potentially truncated if the env terminated early)
        with four added columns: ``reward``, ``trial_idx``, ``state_type``,
        ``return_to_go``.
    """
    env = env_factory()
    seq = actions[action_col].to_numpy()

    rewards: list[float] = []
    trial_ids: list[int] = []
    state_types: list[int] = []

    old_stdout = sys.stdout
    if suppress_env_stdout:
        sys.stdout = io.StringIO()
    try:
        env.reset(seed=seed)
        for a in seq:
            # Capture trial_count and state_type *before* step is processed.
            cycle_id = int(env.trial_count)
            state_type = _read_state_type(env)
            _, r, term, trunc, _ = env.step(int(a))
            rewards.append(float(r))
            trial_ids.append(cycle_id)
            state_types.append(state_type)
            if term or trunc:
                break
    finally:
        if suppress_env_stdout:
            sys.stdout = old_stdout
        env.close()

    n = len(rewards)
    out = actions.iloc[:n].copy()
    out[REWARD_COL] = np.asarray(rewards, dtype=np.float32)
    out[TRIAL_IDX_COL] = np.asarray(trial_ids, dtype=np.int32)
    out[STATE_TYPE_COL] = np.asarray(state_types, dtype=np.int32)
    out[RTG_COL] = per_cycle_suffix_sum(out[REWARD_COL].to_numpy(),
                                         out[TRIAL_IDX_COL].to_numpy())
    return out
