"""Golden-output regression tests for CornerMazeEnv.step().

Ported from legacy ``tests/test_step_regression.py`` with imports updated
for the new package path.

Goldens were *recorded against this port at seed=42* to lock in current
behavior. (The action goldens come from legacy verbatim; the reward
goldens in legacy were stale — recorded under an older constants set
where STEP_FORWARD_COST also equalled -0.001, then never updated when
forward cost was reduced to -0.0005. They fail against legacy too.) The
action goldens passing here is the load-bearing port-validity check,
because action selection is fully driven by env state (action_mask).
"""
from __future__ import annotations

import io
import random
import sys

import numpy as np
import pytest

from corner_maze_rl.env.corner_maze_env import CornerMazeEnv


def _run_episode(session_type, orientation, goal_loc, seed, max_steps=80):
    """Run an episode with deterministic mask-based action selection.

    Uses a separate RNG (seed+1) to pick among valid actions at each step
    so action selection is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = CornerMazeEnv(
        session_type=session_type,
        agent_cue_goal_orientation=orientation,
        start_goal_location=goal_loc,
    )

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()  # silence env's per-step print
    try:
        env.reset(seed=seed)

        rng = random.Random(seed + 1)
        rewards = []
        terminateds = []
        truncateds = []
        actions_taken = []
        for _ in range(max_steps):
            mask = env.get_action_mask()
            mask[4] = False  # disable pause
            valid = [j for j, v in enumerate(mask) if v]
            if not valid:
                valid = [0]
            action = valid[rng.randint(0, len(valid) - 1)]
            actions_taken.append(action)
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(round(reward, 6))
            terminateds.append(terminated)
            truncateds.append(truncated)
            if terminated or truncated:
                break
    finally:
        sys.stdout = old_stdout
        env.close()

    return actions_taken, rewards, terminateds, truncateds


# Golden outputs recorded with seed=42, max_steps=80 against the legacy env.
GOLDEN_SINGLE_TRIAL_ACTIONS = [
    0, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 2,
    1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 1, 2, 2, 1, 2, 0, 1,
    0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2,
    1, 2, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 1, 0, 2, 1, 1, 2, 0, 2,
]
GOLDEN_SINGLE_TRIAL_REWARDS = [
    -0.001, -0.001, -0.0005, -0.001, -0.001, -0.001, -0.0005, -0.0005,
    -0.001, -0.001, -0.0005, -0.001, -0.0005, -0.001, -0.0005, -0.001,
    -0.0005, -0.001, -0.0005, -0.0005, -0.001, -0.001, -0.001, -0.001,
    -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.0005, -0.001,
    -0.001, -0.001, -0.0005, -0.0005, -0.001, -0.0005, -0.001, -0.001,
    -0.001, -0.001, -0.0005, -0.001, -0.001, -0.001, -0.001, -0.001,
    -0.001, -0.001, -0.0005, -0.0005, -0.001, -0.0005, -0.0005, -0.0005,
    -0.0005, -0.001, -0.001, -0.0005, -0.001, -0.0005, -0.0005, -0.001,
    -0.001, -0.0005, -0.0005, -0.0005, -0.001, -0.001, -0.001, -0.001,
    -0.001, -0.001, -0.0005, -0.001, -0.001, -0.0005, -0.001, -0.0005,
]
GOLDEN_EXPOSURE_ACTIONS = list(GOLDEN_SINGLE_TRIAL_ACTIONS)
GOLDEN_EXPOSURE_REWARDS = list(GOLDEN_SINGLE_TRIAL_REWARDS)


# ---------------------------------------------------------------------------
# Single-trial paradigm
# ---------------------------------------------------------------------------

def test_single_trial_actions_deterministic():
    actions, _, _, _ = _run_episode("PI+VC f2 single trial", "N/NE", "NE", seed=42)
    assert actions == GOLDEN_SINGLE_TRIAL_ACTIONS


def test_single_trial_rewards_match():
    _, rewards, _, _ = _run_episode("PI+VC f2 single trial", "N/NE", "NE", seed=42)
    assert rewards == GOLDEN_SINGLE_TRIAL_REWARDS


def test_single_trial_no_termination():
    _, _, terminateds, truncateds = _run_episode(
        "PI+VC f2 single trial", "N/NE", "NE", seed=42
    )
    assert not any(terminateds)
    assert not any(truncateds)


# ---------------------------------------------------------------------------
# Exposure paradigm
# ---------------------------------------------------------------------------

def test_exposure_actions_deterministic():
    actions, _, _, _ = _run_episode("exposure", "N/NE", "NE", seed=42)
    assert actions == GOLDEN_EXPOSURE_ACTIONS


def test_exposure_rewards_match():
    _, rewards, _, _ = _run_episode("exposure", "N/NE", "NE", seed=42)
    assert rewards == GOLDEN_EXPOSURE_REWARDS


def test_exposure_no_termination_in_80_steps():
    _, _, terminateds, truncateds = _run_episode("exposure", "N/NE", "NE", seed=42)
    assert not any(terminateds)
    assert not any(truncateds)


# ---------------------------------------------------------------------------
# Reset / obs shape
# ---------------------------------------------------------------------------

def test_reset_produces_valid_obs():
    random.seed(42)
    np.random.seed(42)
    env = CornerMazeEnv(
        session_type="PI+VC f2 single trial",
        agent_cue_goal_orientation="N/NE",
        start_goal_location="NE",
    )
    obs, _ = env.reset(seed=42)
    assert "image" in obs
    assert "direction" in obs
    assert "mission" in obs
    assert obs["image"].shape == (21, 21, 3)
    env.close()
