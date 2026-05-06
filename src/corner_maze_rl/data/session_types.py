"""Mapping from yoked dataset metadata to ``CornerMazeEnv`` constructor kwargs.

Per plan §9, each (training_group, yoked_session_type) pair maps to a
specific env paradigm. This module owns that table.

Two cells are intentionally unmapped (plan §13 TODO):
  * ``Fixed Cue 1 Twist`` — all groups
  * ``VC × Dark Train``

For any unmapped pair, ``map_session_to_env_kwargs`` returns ``None`` so the
caller can skip with a warning rather than silently using a wrong paradigm.
"""
from __future__ import annotations

from typing import Mapping


# (training_group, yoked_session_type) → env paradigm string
PARADIGM_MAP: Mapping[tuple[str, str], str] = {
    # PI+VC
    ("PI+VC", "Rotate Train"):   "PI+VC f2 rotate",
    ("PI+VC", "Fixed Cue 1"):    "PI+VC f2 novel route",
    ("PI+VC", "Dark Train"):     "PI+VC f2 no cue",
    # PI
    ("PI",    "Fixed Cue 1"):    "PI novel route cue",
    ("PI",    "Dark Train"):     "PI acquisition",
    # VC
    ("VC",    "Rotate Train"):   "VC acquisition",
    ("VC",    "Fixed Cue 1"):    "VC novel route fixed",
    # PI+VC_f1: not yet yoked (plan §9.2 marks NotImplementedError until data is added).
    # VC_DREADDs: out of scope for student build.
}


# Subjects in the f1 cohort that need yoking-pipeline extension before they
# can be used. Plan §9.2.
F1_SUBJECT_NAMES: frozenset[str] = frozenset(
    {"CM057", "CM058", "CM059", "CM060", "CM061", "CM063", "CM064"}
)


# Per-group session-arc ordering. Plan §13 ⏳: unresolved. Until the user
# specifies the canonical sequence per group, the runner should consume
# whatever ordering the yoked sessions table provides (sorted by
# session_number). This dict reserves the API surface.
SESSION_SEQUENCES: Mapping[str, list[str] | None] = {
    "pi_vc":    None,  # TODO: lock canonical order
    "pi":       None,
    "vc":       None,
    "pi_vc_f1": None,  # raises NotImplementedError per plan §9.2
}


def map_session_to_env_kwargs(
    *,
    training_group: str,
    yoked_session_type: str,
    cue_goal_orientation: str,
    start_goal_location: str | None = None,
    obs_mode: str = "view",
) -> dict | None:
    """Build kwargs for ``CornerMazeEnv(...)`` for a yoked session.

    Returns ``None`` if the (group, session_type) pair is unmapped (TODO
    cells per plan §13). Caller decides whether to skip the session.
    """
    paradigm = PARADIGM_MAP.get((training_group, yoked_session_type))
    if paradigm is None:
        return None
    return {
        "session_type": paradigm,
        "agent_cue_goal_orientation": cue_goal_orientation,
        "start_goal_location": start_goal_location,
        "obs_mode": obs_mode,
    }


def is_unmapped(training_group: str, yoked_session_type: str) -> bool:
    """Convenience predicate matching ``map_session_to_env_kwargs is None``."""
    return (training_group, yoked_session_type) not in PARADIGM_MAP


def assert_subject_group_match(
    training_group: str,
    subject_training_group: str,
) -> None:
    """Enforce plan §9.1.1: subject's group must match the chosen group.

    Used by the runner / CLI to fail fast at config-validation time. Raises
    ``ValueError`` with a clear message on mismatch.
    """
    if training_group != subject_training_group:
        raise ValueError(
            f"Subject is in training_group {subject_training_group!r}; "
            f"cannot run sequence for training_group {training_group!r}. "
            "Pick a subject from the matching group's roster."
        )
