"""Tests for corner_maze_rl.data.canonical.

Covers:
- Rotation primitives (compass + xy + direction).
- ``canonicalize_session`` identity / round-trip / no-op cases.
- Spot checks against env constants: rotated arm / cue / goal indices
  match the actual env layout positions after rotation.
- Pass-through for multi-cue (rotate-probe / VC) sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corner_maze_rl.data.canonical import (
    canonicalize_session,
    compass_rotation_for_cue,
    is_canonicalizable,
    rotate_actions,
    rotate_direction,
    rotate_trial_configs,
    rotate_xy,
    session_cue_indices,
)
from corner_maze_rl.env.constants import CUE_LOCATIONS, WELL_LOCATIONS


# ---------------------------------------------------------------------------
# compass_rotation_for_cue: identity when cue is already north
# ---------------------------------------------------------------------------

def test_R_is_zero_when_cue_is_north():
    # cue_idx=3 → CUE_LOCATIONS[3] = (6, 1) = north arm
    assert compass_rotation_for_cue(3) == 0


def test_R_for_east_cue_is_one():
    assert compass_rotation_for_cue(0) == 1


def test_R_for_south_cue_is_two():
    assert compass_rotation_for_cue(1) == 2


def test_R_for_west_cue_is_three():
    assert compass_rotation_for_cue(2) == 3


# ---------------------------------------------------------------------------
# rotate_xy: ground-truth against env constants
# ---------------------------------------------------------------------------

def test_rotate_xy_R0_is_identity():
    assert rotate_xy(5, 7, 0) == (5, 7)


def test_rotate_xy_east_cue_rotates_to_north():
    # CUE_LOCATIONS[0] = (11, 6) = east arm. R=1 (1 CCW) should land it at
    # CUE_LOCATIONS[3] = (6, 1) = north arm.
    east = CUE_LOCATIONS[0]
    north = CUE_LOCATIONS[3]
    assert rotate_xy(*east, 1) == north


def test_rotate_xy_south_to_north_takes_two_steps():
    south = CUE_LOCATIONS[1]
    north = CUE_LOCATIONS[3]
    assert rotate_xy(*south, 2) == north


def test_rotate_xy_west_to_north_takes_three_steps():
    west = CUE_LOCATIONS[2]
    north = CUE_LOCATIONS[3]
    assert rotate_xy(*west, 3) == north


def test_rotate_xy_four_rotations_is_identity():
    for x, y in [(0, 0), (5, 7), (11, 11), (6, 6)]:
        assert rotate_xy(x, y, 4) == (x, y)


def test_rotate_xy_well_NE_to_NW_under_one_CCW():
    # WELL_LOCATIONS = [(11, 11) SE, (1, 11) SW, (1, 1) NW, (11, 1) NE]
    # NE corner = (11, 1). 90° CCW around (6,6) → (1, 1) = NW. ✓
    assert rotate_xy(11, 1, 1) == (1, 1)


# ---------------------------------------------------------------------------
# rotate_direction
# ---------------------------------------------------------------------------

def test_rotate_direction_R0_is_identity():
    for d in range(4):
        assert int(rotate_direction(d, 0)) == d


def test_rotate_direction_one_ccw():
    # MiniGrid dir enum: 0=E, 1=S, 2=W, 3=N. CCW one step: E→N, S→E, W→S, N→W
    # New direction = (d - R) mod 4. R=1: E(0)→3(N), S(1)→0(E), W(2)→1(S), N(3)→2(W)
    assert int(rotate_direction(0, 1)) == 3  # E → N
    assert int(rotate_direction(1, 1)) == 0  # S → E
    assert int(rotate_direction(2, 1)) == 1  # W → S
    assert int(rotate_direction(3, 1)) == 2  # N → W


def test_rotate_direction_vector():
    arr = np.array([0, 1, 2, 3])
    out = rotate_direction(arr, 1)
    np.testing.assert_array_equal(out, np.array([3, 0, 1, 2]))


# ---------------------------------------------------------------------------
# is_canonicalizable / session_cue_indices
# ---------------------------------------------------------------------------

def test_constant_cue_session_is_canonicalizable():
    tc = [[1, 0, 0, "trained"], [3, 0, 1, "trained"], [1, 0, 2, "trained"]]
    assert is_canonicalizable(tc) is True


def test_multi_cue_session_is_not_canonicalizable():
    tc = [[1, 0, 0, "trained"], [3, 1, 1, "trained"], [1, 2, 2, "trained"]]
    assert is_canonicalizable(tc) is False


def test_empty_trial_configs_is_not_canonicalizable():
    assert is_canonicalizable([]) is False


def test_session_cue_indices_returns_set():
    tc = [[1, 0, 0, "trained"], [3, 0, 1, "trained"]]
    assert session_cue_indices(tc) == {0}


# ---------------------------------------------------------------------------
# rotate_trial_configs: cue lands at north when R was chosen from it
# ---------------------------------------------------------------------------

def test_rotate_trial_configs_lands_cue_at_north():
    # CM000-style: arm=E(1), cue=E(0), goal=NE(0). R=1.
    tc = [[1, 0, 0, "trained"], [3, 0, 1, "trained"]]
    R = compass_rotation_for_cue(0)
    out = rotate_trial_configs(tc, R)
    # Every cue should be 3 (north)
    assert all(t[1] == 3 for t in out)
    # First trial: arm E→N, goal NE→NW
    assert out[0] == [0, 3, 3, "trained"]
    # Second trial: arm W→S, goal SE→NE
    assert out[1] == [2, 3, 0, "trained"]


def test_rotate_trial_configs_R0_is_identity():
    tc = [[0, 3, 0, "trained"], [2, 3, 2, "trained"]]
    out = rotate_trial_configs(tc, 0)
    # Same content, but new list-of-lists (caller-safe copy)
    assert out == [[0, 3, 0, "trained"], [2, 3, 2, "trained"]]


def test_rotate_trial_configs_four_rotations_is_identity():
    tc = [[1, 0, 0, "trained"], [3, 2, 1, "probe_trained"]]
    rotated = rotate_trial_configs(tc, 4)
    assert rotated == [[1, 0, 0, "trained"], [3, 2, 1, "probe_trained"]]


# ---------------------------------------------------------------------------
# rotate_actions: vectorized rotation of (grid_x, grid_y, direction)
# ---------------------------------------------------------------------------

def _toy_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1],
            "step": [0, 1, 2],
            "action": [2, 1, 2],
            "grid_x": [11, 6, 9],
            "grid_y": [6, 6, 6],
            "direction": [2, 3, 0],
            "rewarded": [0, 0, 1],
        }
    )


def test_rotate_actions_R0_returns_copy_unchanged():
    df = _toy_actions()
    out = rotate_actions(df, 0)
    pd.testing.assert_frame_equal(out, df)
    # Sanity: it's a copy, not a view (mutating out shouldn't touch df)
    out.loc[0, "grid_x"] = 999
    assert df.loc[0, "grid_x"] == 11


def test_rotate_actions_rotates_columns_consistently():
    df = _toy_actions()
    out = rotate_actions(df, 1)  # one CCW
    # (11, 6) → (6, 1)
    assert (int(out.loc[0, "grid_x"]), int(out.loc[0, "grid_y"])) == (6, 1)
    # direction 2 (W) → (2-1) mod 4 = 1 (S)
    assert int(out.loc[0, "direction"]) == 1
    # Action and rewarded unchanged
    assert list(out["action"]) == [2, 1, 2]
    assert list(out["rewarded"]) == [0, 0, 1]


def test_rotate_actions_four_rotations_is_identity():
    df = _toy_actions()
    out = rotate_actions(df, 4)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_rotate_actions_preserves_extra_columns():
    df = _toy_actions()
    df["extra"] = ["a", "b", "c"]
    out = rotate_actions(df, 1)
    assert list(out["extra"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# canonicalize_session: end-to-end
# ---------------------------------------------------------------------------

def test_canonicalize_session_constant_cue_returns_nonzero_R():
    tc = [[1, 0, 0, "trained"], [3, 0, 1, "trained"]]
    df = _toy_actions()
    rotated_tc, rotated_df, R = canonicalize_session(tc, df)
    assert R == 1
    assert all(t[1] == 3 for t in rotated_tc)
    # grid_x was 11 → 6, grid_y was 6 → 1 (1 CCW around (6,6))
    assert (int(rotated_df.loc[0, "grid_x"]), int(rotated_df.loc[0, "grid_y"])) == (6, 1)


def test_canonicalize_session_cue_already_north_is_no_op():
    tc = [[0, 3, 0, "trained"], [2, 3, 2, "trained"]]
    df = _toy_actions()
    rotated_tc, rotated_df, R = canonicalize_session(tc, df)
    assert R == 0
    assert rotated_tc == tc
    pd.testing.assert_frame_equal(rotated_df, df)


def test_canonicalize_session_multi_cue_passes_through():
    # Rotate-probe / VC style: cues vary across trials.
    tc = [[1, 0, 0, "trained"], [3, 1, 1, "probe_trained"]]
    df = _toy_actions()
    rotated_tc, rotated_df, R = canonicalize_session(tc, df)
    assert R == 0
    assert rotated_tc == list(tc)
    pd.testing.assert_frame_equal(rotated_df, df)


def test_canonicalize_session_round_trip():
    """Rotating by R and then by 4-R returns the original (for constant-cue)."""
    tc = [[1, 0, 0, "trained"], [3, 0, 1, "trained"]]
    df = _toy_actions()
    rotated_tc, rotated_df, R = canonicalize_session(tc, df)
    back_tc = rotate_trial_configs(rotated_tc, 4 - R)
    back_df = rotate_actions(rotated_df, 4 - R)
    assert back_tc == [list(t) for t in tc]
    pd.testing.assert_frame_equal(
        back_df.reset_index(drop=True), df.reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Cross-check: applying R via the helper to the cue arm's coordinate
# always lands at the north cue position
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cue_idx", [0, 1, 2, 3])
def test_cue_arm_coordinate_rotates_to_north(cue_idx):
    R = compass_rotation_for_cue(cue_idx)
    cx, cy = CUE_LOCATIONS[cue_idx]
    new_x, new_y = rotate_xy(cx, cy, R)
    assert (new_x, new_y) == CUE_LOCATIONS[3]


# ---------------------------------------------------------------------------
# End-to-end: rotated trial_configs + rotated actions replay through env
# with zero divergence. Skipped if the dataset isn't on disk.
# ---------------------------------------------------------------------------

import json
from pathlib import Path


_DATASET_DIR = Path("data/yoked/dataset")


def _dataset_available() -> bool:
    return (_DATASET_DIR / "subjects.parquet").is_file()


@pytest.mark.skipif(not _dataset_available(), reason="yoked dataset not on disk")
@pytest.mark.parametrize(
    "training_group,session_type",
    [
        ("PI", "Dark Train"),
        ("PI", "Fixed Cue 1"),
        ("PI+VC", "Fixed Cue 1"),
        ("PI+VC_f1", "Fixed Cue 1 Twist"),
    ],
)
def test_env_replay_parity_after_canonicalization(training_group, session_type):
    """Rotated trial_configs + rotated actions replayed through env produce
    zero (env.agent_pos / agent_dir) divergence from the rotated dataset
    (grid_x, grid_y, direction). One canonicalizable cell per (group,
    session_type) pair currently in PARADIGM_MAP."""
    from corner_maze_rl.data.load import (
        YokedPaths,
        load_actions_for_session,
        load_sessions,
        load_subjects,
    )
    from corner_maze_rl.data.session_types import (
        PARADIGM_MAP,
        map_session_to_env_kwargs,
    )
    from corner_maze_rl.env.corner_maze_env import CornerMazeEnv

    paths = YokedPaths.from_dir(_DATASET_DIR, actions_variant="synthetic_pretrial")
    subjects = load_subjects(paths)
    sessions = load_sessions(paths)
    acq = sessions[sessions["session_phase"] == "Acquisition"].merge(
        subjects[["subject_id", "subject_name", "training_group", "cue_goal_orientation"]],
        on="subject_id", suffixes=("", "_subj"),
    )
    sub = acq[
        (acq["training_group"] == training_group)
        & (acq["session_type"] == session_type)
    ].sort_values(["subject_name", "session_number"])
    if sub.empty:
        pytest.skip(f"no sessions for ({training_group}, {session_type}) in dataset")
    row = sub.iloc[0]
    tc = json.loads(row["trial_configs"])
    actions = load_actions_for_session(paths, int(row["session_id"]))

    rotated_tc, rotated_actions, R = canonicalize_session(tc, actions)

    goal_idx_to_dir = {0: "NE", 1: "SE", 2: "SW", 3: "NW"}
    first_goal = (
        goal_idx_to_dir.get(int(rotated_tc[0][2])) if rotated_tc else None
    )
    kwargs = map_session_to_env_kwargs(
        training_group=row["training_group"],
        yoked_session_type=row["session_type"],
        cue_goal_orientation=row["cue_goal_orientation"],
        start_goal_location=first_goal,
        trial_configs=rotated_tc,
    )
    assert (training_group, session_type) in PARADIGM_MAP
    env = CornerMazeEnv(render_mode="rgb_array", max_steps=10000, **kwargs)
    env.reset(seed=int(row["seed"]))

    div = 0
    first = None
    for i in range(len(rotated_actions)):
        r = rotated_actions.iloc[i]
        ex, ey = env.agent_pos
        ed = env.agent_dir
        rx, ry, rd = int(r["grid_x"]), int(r["grid_y"]), int(r["direction"])
        if (ex, ey, ed) != (rx, ry, rd):
            div += 1
            if first is None:
                first = (i, (ex, ey, ed), (rx, ry, rd))
        env.step(int(r["action"]))
    assert div == 0, (
        f"{training_group}/{session_type} {row['subject_name']} "
        f"sess {int(row['session_number'])} R={R}: {div} divergences; first={first}"
    )


@pytest.mark.skipif(not _dataset_available(), reason="yoked dataset not on disk")
def test_consume_side_alignment_for_training():
    """Simulates the future yoked-training rollout. Mode 2 (pre-rotated on
    disk): rotate trial_configs + actions, replay, then forget the
    rotated trial_configs (as if reading a pre-canonicalized parquet).
    Re-derive rotated_tc from raw_tc + R, rebuild env, replay rotated
    actions. Must produce zero divergence. This is the contract the
    yoked-training loop will rely on."""
    from corner_maze_rl.data.load import (
        YokedPaths,
        load_actions_for_session,
        load_sessions,
        load_subjects,
    )
    from corner_maze_rl.data.session_types import map_session_to_env_kwargs
    from corner_maze_rl.env.corner_maze_env import CornerMazeEnv

    paths = YokedPaths.from_dir(_DATASET_DIR, actions_variant="synthetic_pretrial")
    subjects = load_subjects(paths)
    sessions = load_sessions(paths)
    acq = sessions[sessions["session_phase"] == "Acquisition"].merge(
        subjects[["subject_id", "subject_name", "training_group", "cue_goal_orientation"]],
        on="subject_id", suffixes=("", "_subj"),
    )
    # Pick the first session per (group, session_type) cell, covering R=0..3.
    seen = set()
    rows_to_check = []
    for _, r in acq.sort_values(["training_group", "session_type", "subject_name", "session_number"]).iterrows():
        key = (r["training_group"], r["session_type"])
        if key in seen:
            continue
        seen.add(key)
        rows_to_check.append(r)

    goal_idx_to_dir = {0: "NE", 1: "SE", 2: "SW", 3: "NW"}
    for row in rows_to_check:
        raw_tc = json.loads(row["trial_configs"])
        if not is_canonicalizable(raw_tc):
            continue  # VC etc. — out of scope here
        # Pre-rotate (build phase)
        actions_raw = load_actions_for_session(paths, int(row["session_id"]))
        rotated_tc_buildside, rotated_actions, R = canonicalize_session(raw_tc, actions_raw)

        # ─── Forget rotated_tc_buildside; simulate consume-side recompute. ───
        rotated_tc_recompute = rotate_trial_configs(raw_tc, R)
        assert rotated_tc_recompute == rotated_tc_buildside, (
            "Mode-2 consume-side recompute must equal build-side rotation"
        )

        first_goal = goal_idx_to_dir.get(int(rotated_tc_recompute[0][2]))
        kwargs = map_session_to_env_kwargs(
            training_group=row["training_group"],
            yoked_session_type=row["session_type"],
            cue_goal_orientation=row["cue_goal_orientation"],
            start_goal_location=first_goal,
            trial_configs=rotated_tc_recompute,
        )
        env = CornerMazeEnv(render_mode="rgb_array", max_steps=10000, **kwargs)
        env.reset(seed=int(row["seed"]))

        div = 0
        for i in range(len(rotated_actions)):
            r = rotated_actions.iloc[i]
            ex, ey = env.agent_pos
            ed = env.agent_dir
            rx, ry, rd = int(r["grid_x"]), int(r["grid_y"]), int(r["direction"])
            if (ex, ey, ed) != (rx, ry, rd):
                div += 1
            env.step(int(r["action"]))
        assert div == 0, (
            f"Consume-side alignment failed for "
            f"{row['training_group']}/{row['session_type']} "
            f"{row['subject_name']} sess {int(row['session_number'])} R={R}: "
            f"{div} divergences"
        )
