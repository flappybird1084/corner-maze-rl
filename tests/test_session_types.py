"""Tests for the exposure entries added to ``PARADIGM_MAP``.

The two ``("PI", "exposure"*)`` rows are what unblocks
``build_returns_dataset.py`` for the rat-neural-model-2 → yoked-form pipeline;
without them every exposure session is silently dropped as "unmapped".
"""
from __future__ import annotations

from corner_maze_rl.data.session_types import (
    PARADIGM_MAP,
    is_unmapped,
    map_session_to_env_kwargs,
)


# ---------------------------------------------------------------------------
# Direct dict access — guard against accidental key removal/rename.
# ---------------------------------------------------------------------------

def test_paradigm_map_pi_exposure_present():
    assert PARADIGM_MAP[("PI", "exposure")] == "exposure"


def test_paradigm_map_pi_exposure_b_present():
    assert PARADIGM_MAP[("PI", "exposure_b")] == "exposure_b"


# ---------------------------------------------------------------------------
# map_session_to_env_kwargs — the function the runner actually calls.
# ---------------------------------------------------------------------------

def test_map_exposure_returns_env_kwargs():
    kw = map_session_to_env_kwargs(
        training_group="PI",
        yoked_session_type="exposure",
        cue_goal_orientation="NS",
    )
    assert kw is not None
    assert kw["session_type"] == "exposure"
    assert kw["agent_cue_goal_orientation"] == "NS"
    assert kw["start_goal_location"] is None
    assert kw["obs_mode"] == "view"


def test_map_exposure_b_returns_env_kwargs():
    kw = map_session_to_env_kwargs(
        training_group="PI",
        yoked_session_type="exposure_b",
        cue_goal_orientation="NS",
    )
    assert kw is not None
    assert kw["session_type"] == "exposure_b"


# ---------------------------------------------------------------------------
# Negative cases — make sure the new rows didn't over-add.
# ---------------------------------------------------------------------------

def test_vc_exposure_not_added():
    """Only PI rows were added; VC × exposure should still be unmapped.

    The converter normalizes everyone to training_group='PI' for exposure
    data, so we never expect (VC, exposure) to be queried — but making
    sure we didn't blanket-add keeps the unmapped-skip behaviour intact
    for any future (group, session_type) combination we haven't vetted.
    """
    assert ("VC", "exposure") not in PARADIGM_MAP
    assert is_unmapped("VC", "exposure")
    assert map_session_to_env_kwargs(
        training_group="VC",
        yoked_session_type="exposure",
        cue_goal_orientation="NS",
    ) is None


def test_existing_paradigm_still_resolves():
    """Sanity: pre-existing entries weren't disturbed."""
    assert PARADIGM_MAP[("PI+VC", "Rotate Train")] == "PI+VC f2 rotate"
    kw = map_session_to_env_kwargs(
        training_group="PI+VC",
        yoked_session_type="Rotate Train",
        cue_goal_orientation="NS",
    )
    assert kw is not None
    assert kw["session_type"] == "PI+VC f2 rotate"
