"""Tests for corner_maze_rl.data.compute_returns and session_types."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corner_maze_rl.data.compute_returns import (
    REWARD_COL,
    RTG_COL,
    STATE_TYPE_COL,
    TRIAL_IDX_COL,
    compute_returns_for_session,
    per_cycle_suffix_sum,
)
from corner_maze_rl.data.session_types import (
    F1_SUBJECT_NAMES,
    PARADIGM_MAP,
    assert_subject_group_match,
    is_unmapped,
    map_session_to_env_kwargs,
)


# ---------------------------------------------------------------------------
# per_cycle_suffix_sum (pure RTG arithmetic)
# ---------------------------------------------------------------------------

def test_suffix_sum_single_cycle():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    cyc = np.array([0, 0, 0, 0])
    rtg = per_cycle_suffix_sum(r, cyc)
    # 1+2+3+4=10, 2+3+4=9, 3+4=7, 4
    assert rtg.tolist() == [10.0, 9.0, 7.0, 4.0]


def test_suffix_sum_resets_at_cycle_boundary():
    r = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
    cyc = np.array([0, 0, 0, 1, 1])
    rtg = per_cycle_suffix_sum(r, cyc)
    # cycle 0: 7, 6, 5; cycle 1: 2, 1
    assert rtg.tolist() == [7.0, 6.0, 5.0, 2.0, 1.0]


def test_suffix_sum_handles_singleton_cycles():
    r = np.array([1.0, 2.0, 3.0])
    cyc = np.array([0, 1, 2])
    rtg = per_cycle_suffix_sum(r, cyc)
    assert rtg.tolist() == [1.0, 2.0, 3.0]


def test_suffix_sum_empty():
    r = np.array([], dtype=np.float32)
    cyc = np.array([], dtype=np.int32)
    rtg = per_cycle_suffix_sum(r, cyc)
    assert rtg.shape == (0,)


def test_suffix_sum_length_mismatch_raises():
    with pytest.raises(ValueError):
        per_cycle_suffix_sum(np.array([1.0]), np.array([0, 0]))


def test_suffix_sum_dtype_is_float32():
    r = np.array([1.0, 2.0], dtype=np.float64)
    cyc = np.array([0, 0])
    rtg = per_cycle_suffix_sum(r, cyc)
    assert rtg.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_returns_for_session (env replay)
# ---------------------------------------------------------------------------

def _env_factory():
    from corner_maze_rl.env.corner_maze_env import CornerMazeEnv
    return CornerMazeEnv(
        session_type="PI+VC f2 single trial",
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
    )


def test_compute_returns_for_session_adds_required_columns():
    actions = pd.DataFrame({
        "action": [2, 2, 0, 1, 2, 2, 1, 2, 0, 2],
        "step": list(range(10)),
    })
    out = compute_returns_for_session(actions, _env_factory, seed=42)
    for col in (REWARD_COL, RTG_COL, TRIAL_IDX_COL, STATE_TYPE_COL):
        assert col in out.columns
    assert len(out) <= len(actions)  # may be shorter if env terminated
    assert out["action"].tolist() == actions["action"].tolist()[: len(out)]


def test_compute_returns_rewards_match_env_step():
    """Rewards in the output must match what env.step would emit for the
    given action sequence."""
    from corner_maze_rl.env.corner_maze_env import CornerMazeEnv
    import io
    import sys

    actions = pd.DataFrame({"action": [2, 0, 1, 2, 2]})
    out = compute_returns_for_session(actions, _env_factory, seed=42)

    # Replay manually for cross-check
    env = CornerMazeEnv(
        session_type="PI+VC f2 single trial",
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
    )
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        env.reset(seed=42)
        manual_rewards = []
        for a in actions["action"]:
            _, r, _, _, _ = env.step(int(a))
            manual_rewards.append(round(r, 6))
    finally:
        sys.stdout = old
        env.close()

    assert [round(x, 6) for x in out[REWARD_COL].tolist()] == manual_rewards


def test_compute_returns_rtg_within_single_cycle():
    """For a short replay that doesn't cross a trial boundary, RTG[t] must
    equal sum(rewards[t:])."""
    actions = pd.DataFrame({"action": [2, 0, 1, 2, 2, 0, 1, 2]})
    out = compute_returns_for_session(actions, _env_factory, seed=42)
    rewards = out[REWARD_COL].to_numpy()
    rtg = out[RTG_COL].to_numpy()
    cyc = out[TRIAL_IDX_COL].to_numpy()
    # In this short run they should all be in the same cycle (trial 0).
    assert (cyc == cyc[0]).all()
    expected = np.cumsum(rewards[::-1])[::-1]
    assert np.allclose(rtg, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# session_types: mapping table
# ---------------------------------------------------------------------------

def test_paradigm_map_known_pairs():
    assert PARADIGM_MAP[("PI+VC", "Rotate Train")] == "PI+VC f2 rotate"
    assert PARADIGM_MAP[("VC", "Rotate Train")] == "VC acquisition"
    assert PARADIGM_MAP[("PI", "Dark Train")] == "PI acquisition"


def test_map_session_to_env_kwargs_known():
    kw = map_session_to_env_kwargs(
        training_group="PI+VC",
        yoked_session_type="Fixed Cue 1",
        cue_goal_orientation="N/NE",
    )
    assert kw is not None
    assert kw["session_type"] == "PI+VC f2 novel route"
    assert kw["agent_cue_goal_orientation"] == "N/NE"


def test_map_session_to_env_kwargs_unmapped_returns_none():
    """Plan §13 TODO cells: Fixed Cue 1 Twist (all groups) and VC × Dark Train."""
    assert map_session_to_env_kwargs(
        training_group="PI+VC",
        yoked_session_type="Fixed Cue 1 Twist",
        cue_goal_orientation="N/NE",
    ) is None
    assert map_session_to_env_kwargs(
        training_group="VC",
        yoked_session_type="Dark Train",
        cue_goal_orientation="N/NE",
    ) is None


def test_is_unmapped():
    assert is_unmapped("VC", "Dark Train") is True
    assert is_unmapped("PI+VC", "Rotate Train") is False


def test_assert_subject_group_match_ok():
    # No raise.
    assert_subject_group_match("PI+VC", "PI+VC")


def test_assert_subject_group_match_mismatch():
    with pytest.raises(ValueError, match="cannot run"):
        assert_subject_group_match("VC", "PI+VC")


def test_f1_cohort_listed():
    assert "CM057" in F1_SUBJECT_NAMES
    assert "CM064" in F1_SUBJECT_NAMES
    # CM005 is PI+VC main cohort, not f1
    assert "CM005" not in F1_SUBJECT_NAMES
