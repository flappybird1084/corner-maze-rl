"""Cue-canonical (north) rotation for yoked sessions.

The corner-maze task has four-fold symmetry: cue at one of four arms,
goal at one of four corners, start arm at one of four arms. Different
rats experienced the cue at different arms — but for training we want
the agent to always see the cue at a canonical position (north arm).
Rotating each rat's data so its cue lands at north pools data across
rats into one canonical task.

Scope (per design discussion 2026-05-10):
  * Canonicalizable: sessions whose cue is constant across all trials —
    PI / PI+VC / PI+VC_f1 Acquisition (Dark Train, Fixed Cue 1,
    Fixed Cue 1 Twist).
  * Pass-through: sessions whose cue varies per trial — VC Rotate Train,
    plus any future PI+VC / PI+VC_f1 rotate-probe data. ``canonicalize_session``
    detects these and returns the inputs unchanged with ``R=0``.

The math is fully determined by ``env/constants.py``:

  * Cue index → compass: 0=E, 1=S, 2=W, **3=N**
    (because ``CUE_LOCATIONS = [(11,6), (6,11), (1,6), (6,1)]``)
  * Arm index → compass: 0=N, 1=E, 2=S, 3=W
    (per the ``PRETRIAL_TRIGGER_POSITIONS`` comment)
  * Goal index → compass corner (clockwise from NE): NE=0, SE=1, SW=2, NW=3
  * Direction enum → compass: 0=E, 1=S, 2=W, 3=N (MiniGrid)

To bring a session's cue to N we rotate the world CCW by
``R = (cue_idx + 1) mod 4``. Then for every field that lives in the
shared compass system:

  * ``cue_idx`` → 3 (always)
  * ``arm_idx`` → ``(arm_idx − R) mod 4``
  * ``goal_idx`` → ``(goal_idx − R) mod 4``
  * ``direction`` → ``(direction − R) mod 4``
  * ``(x, y)`` → apply ``(x, y) ↦ (y, 12 − x)`` exactly ``R`` times
    (90° CCW around the grid centre ``(6, 6)`` on a 13×13 grid)
  * ``action`` — unchanged (egocentric)
  * ``rewarded`` — unchanged

Yoked training alignment contract
---------------------------------

The full chain — yoking pipeline → action stream → env replay → training
— stays aligned as long as ``trial_configs`` and the positions in the
action stream are rotated by the **same R**.

There are two valid storage modes:

1. **Raw on disk, canonicalize at consume time** (default):
   ``actions_*.parquet`` holds the rat's original coords, ``sessions.parquet``
   holds original ``trial_configs``. A consumer (the 02 notebook with the
   checkbox, or a future training rollout) calls
   ``canonicalize_session(trial_configs, actions_df)`` to rotate both
   together, then passes the rotated ``trial_configs`` into the env
   kwargs.

2. **Pre-rotated on disk** (``build_returns_dataset.py --canonicalize``):
   ``actions_with_returns.parquet`` has rotated ``grid_x / grid_y /
   direction`` plus a ``canonical_R`` column per row. The rotated
   ``trial_configs`` are **not** persisted — consume-side, you rebuild
   them with ``rotate_trial_configs(raw_tc, R)`` (where ``raw_tc`` comes
   from ``sessions.parquet`` and ``R`` from the parquet's
   ``canonical_R`` column). This is a zero-cost recompute.

Consume-side recipe (mode 2)::

    R = int(actions_with_returns_df["canonical_R"].iloc[0])
    raw_tc = json.loads(sessions_row["trial_configs"])
    rotated_tc = rotate_trial_configs(raw_tc, R)
    env_kwargs = map_session_to_env_kwargs(
        ...,
        trial_configs=rotated_tc,
        start_goal_location=GOAL_IDX_TO_DIR[int(rotated_tc[0][2])],
    )

In either mode, the env-replay parity invariant holds: env built with
``rotated_tc`` produces ``(agent_pos, agent_dir)`` matching the rotated
``(grid_x, grid_y, direction)`` at every step. Tests in
``tests/test_canonical.py`` enforce both directions of this chain.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Hard-coded for the 13×13 corner maze. Rotation is around (6, 6).
_GRID_MAX = 12  # so (x, y) ↦ (y, GRID_MAX − x) for 90° CCW


def compass_rotation_for_cue(cue_idx: int) -> int:
    """Number of 90° CCW rotations needed to bring cue ``cue_idx`` to north.

    Identity for ``cue_idx = 3``. Returns ``R`` in ``{0, 1, 2, 3}``.
    """
    return (int(cue_idx) + 1) % 4


def rotate_xy(x: int | np.ndarray, y: int | np.ndarray, R: int) -> tuple:
    """Apply ``R`` 90°-CCW rotations to ``(x, y)`` around the grid centre.

    Works on scalars or numpy arrays element-wise. Each rotation is
    ``(x, y) ↦ (y, 12 − x)``.
    """
    R = int(R) % 4
    for _ in range(R):
        x, y = y, _GRID_MAX - x
    return x, y


def rotate_direction(direction: int | np.ndarray, R: int) -> int | np.ndarray:
    """Apply ``R`` 90°-CCW rotations to a MiniGrid direction enum."""
    return (np.asarray(direction) - int(R)) % 4


def session_cue_indices(trial_configs: Sequence[Sequence]) -> set[int]:
    """Distinct cue values appearing across a session's trial_configs."""
    return {int(t[1]) for t in trial_configs}


def is_canonicalizable(trial_configs: Sequence[Sequence]) -> bool:
    """True iff the session has exactly one cue across all trials.

    Rotate-probe / VC-style sessions return False — those vary cue per
    trial and would need per-cycle rotation, which we don't do here.
    Empty trial_configs (Exposure) → False (no cue concept).
    """
    if not trial_configs:
        return False
    return len(session_cue_indices(trial_configs)) == 1


def rotate_trial_configs(
    trial_configs: Sequence[Sequence],
    R: int,
) -> list[list]:
    """Apply rotation ``R`` to every ``(arm, cue, goal, tag)`` trial.

    Returns a fresh list of 4-element lists with cue set to ``3 if R fully
    canonicalizes`` (i.e. the input was constant-cue and ``R`` was derived
    from that cue) — otherwise the cue is rotated like any other compass
    field. Callers that bypass ``canonicalize_session`` are responsible
    for passing a coherent ``R``.
    """
    R = int(R) % 4
    out = []
    for t in trial_configs:
        arm, cue, goal, tag = int(t[0]), int(t[1]), int(t[2]), t[3]
        # Arm: 0=N, 1=E, 2=S, 3=W → CCW shift by R
        new_arm = (arm - R) % 4
        # Cue: 0=E, 1=S, 2=W, 3=N → CCW shift by R
        new_cue = (cue - R) % 4
        # Goal: NE=0, SE=1, SW=2, NW=3 → CCW shift by R
        new_goal = (goal - R) % 4
        out.append([new_arm, new_cue, new_goal, tag])
    return out


def rotate_actions(df: pd.DataFrame, R: int) -> pd.DataFrame:
    """Return a copy of ``df`` with ``grid_x / grid_y / direction`` rotated.

    Columns ``action`` and ``rewarded`` are passed through unchanged
    (actions are egocentric; reward credit doesn't depend on coords).
    All other columns are preserved.
    """
    R = int(R) % 4
    out = df.copy()
    if R == 0:
        return out
    x = out["grid_x"].to_numpy()
    y = out["grid_y"].to_numpy()
    new_x, new_y = rotate_xy(x, y, R)
    out["grid_x"] = np.asarray(new_x, dtype=df["grid_x"].dtype)
    out["grid_y"] = np.asarray(new_y, dtype=df["grid_y"].dtype)
    out["direction"] = rotate_direction(out["direction"].to_numpy(), R).astype(
        df["direction"].dtype
    )
    return out


def canonicalize_session(
    trial_configs: Sequence[Sequence],
    actions_df: pd.DataFrame,
) -> tuple[list[list], pd.DataFrame, int]:
    """Top-level canonicalizer.

    Returns ``(rotated_trial_configs, rotated_actions, R)``. For
    multi-cue (non-canonicalizable) sessions returns the inputs
    unchanged with ``R=0`` — callers can detect the no-op via the
    returned ``R``.
    """
    if not is_canonicalizable(trial_configs):
        return list(trial_configs), actions_df, 0
    cue = next(iter(session_cue_indices(trial_configs)))
    R = compass_rotation_for_cue(cue)
    if R == 0:
        return list(trial_configs), actions_df, 0
    return (
        rotate_trial_configs(trial_configs, R),
        rotate_actions(actions_df, R),
        R,
    )


__all__ = [
    "compass_rotation_for_cue",
    "rotate_xy",
    "rotate_direction",
    "session_cue_indices",
    "is_canonicalizable",
    "rotate_trial_configs",
    "rotate_actions",
    "canonicalize_session",
]
