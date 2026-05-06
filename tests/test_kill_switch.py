"""Tests for corner_maze_rl.train.kill_switch.

Covers the canonical curve cases listed in plan §8.2 plus the slope helper.
"""
from __future__ import annotations

import pytest

from corner_maze_rl.train.kill_switch import (
    DEFAULT_CONFIG,
    Decision,
    KillSwitchConfig,
    decide,
    killed_at_payload,
    linear_regression_slope,
)


# ---------------------------------------------------------------------------
# linear_regression_slope
# ---------------------------------------------------------------------------

def test_slope_perfect_line():
    # y = 2t + 1, t = 0..4
    vals = [1, 3, 5, 7, 9]
    assert linear_regression_slope(vals) == pytest.approx(2.0)


def test_slope_flat_zero():
    assert linear_regression_slope([3, 3, 3, 3]) == pytest.approx(0.0)


def test_slope_negative():
    assert linear_regression_slope([10, 8, 6, 4, 2]) == pytest.approx(-2.0)


def test_slope_short_series():
    assert linear_regression_slope([]) == 0.0
    assert linear_regression_slope([5]) == 0.0


# ---------------------------------------------------------------------------
# decide() — canonical curves from plan §8.2
# ---------------------------------------------------------------------------

def test_warmup_continue_below_threshold():
    """No matter the shape, n < WARMUP returns CONTINUE."""
    for shape in ([0, 0, 0], [10, 20, 30], [0] * 9, [32] * 5):
        r = decide(shape)
        assert r.decision is Decision.CONTINUE
        assert r.reason == "warmup"


def test_dead_kills_after_window_of_zeros():
    """All zeros for >= DEAD_WINDOW sessions → KILL_DEAD.

    DEAD_WINDOW=8, WARMUP=10, so we need at least 10 sessions of all zeros
    for the dead check to fire after warmup.
    """
    scores = [0] * 12
    r = decide(scores)
    assert r.decision is Decision.KILL_DEAD
    assert r.n_sessions == 12


def test_dead_does_not_fire_with_single_recent_success():
    """Even one nonzero in last DEAD_WINDOW prevents KILL_DEAD.

    But the run will likely still hit KILL_FLAT — that's a separate path.
    """
    # 12 sessions, but a 1 inside the dead window
    scores = [0] * 8 + [1] + [0] * 3
    r = decide(scores)
    assert r.decision is not Decision.KILL_DEAD


def test_flat_low_kill():
    """Constant low value over many sessions → KILL_FLAT once n ≥ WARMUP+SLOPE_WINDOW."""
    scores = [2] * 20
    r = decide(scores)
    assert r.decision is Decision.KILL_FLAT
    assert r.recent_mean == pytest.approx(2.0)
    assert r.slope == pytest.approx(0.0)


def test_slow_creep_low_continues():
    """Slow upward creep (slope ≈ 0.33) with mean < floor → CONTINUE.

    Goes from 2 → 5 over the slope window. Slope ≈ 0.33 > FLAT_SLOPE_EPS=0.05.
    Floor is 4, mean ≈ 3.5 < 4, but slope rule saves it.
    """
    scores = [2, 2, 3, 3, 3, 4, 4, 4, 5, 5]  # 10 sessions, slope ≈ 0.33
    # pad pre-warmup with zeros so we're past warmup, but the slope-window
    # is the last 10
    full = [0] * 5 + scores  # 15 total, last 10 = scores
    r = decide(full)
    assert r.decision is Decision.CONTINUE
    assert r.slope is not None and r.slope > DEFAULT_CONFIG.flat_slope_eps


def test_fast_creep_low_continues():
    """Sharp upward learning curve from low base → CONTINUE."""
    scores = [0, 0, 1, 2, 4, 6, 8, 10, 11, 12]  # last 10
    full = [0] * 5 + scores
    r = decide(full)
    assert r.decision is Decision.CONTINUE
    assert r.slope is not None and r.slope > DEFAULT_CONFIG.flat_slope_eps


def test_plateau_high_criterion_met():
    """Plateau at high value (mean ≥ CRITERION_MEAN) → CRITERION_MET."""
    scores = [28] * 15
    r = decide(scores)
    assert r.decision is Decision.CRITERION_MET
    assert r.recent_mean is not None and r.recent_mean >= DEFAULT_CONFIG.criterion_mean


def test_plateau_high_criterion_met_takes_priority_over_flat():
    """A flat plateau at 28 should be CRITERION_MET, not KILL_FLAT.

    This is the load-bearing reason for the ABSOLUTE_FLOOR gate: high-mean
    saturation has slope ≈ 0 but is success, not failure.
    """
    scores = [28] * 25
    r = decide(scores)
    assert r.decision is Decision.CRITERION_MET


def test_declining_kills():
    """Negative-slope curve with low mean → KILL_FLAT (asymmetric `<`).

    Need n ≥ WARMUP+SLOPE_WINDOW=20 for the slope check to be eligible.
    Last 10 sessions: mean = 3.5, slope < 0 → kill.
    """
    last10 = [8, 7, 5, 4, 3, 2, 2, 2, 1, 1]
    full = [5] * 10 + last10  # 20 total, slope check now eligible
    r = decide(full)
    assert r.decision is Decision.KILL_FLAT
    assert r.slope is not None and r.slope < 0


def test_noisy_flat_zero_mean_kills_after_warmup_plus_window():
    """Noisy zero-mean signal still gets killed: slope ≈ 0, mean low."""
    # 0/1 alternating, 20 sessions. Mean = 0.5, slope ≈ 0.
    scores = [0, 1] * 10
    r = decide(scores)
    assert r.decision is Decision.KILL_FLAT


def test_hard_cap_kill():
    """Even if creeping, n ≥ HARD_CAP triggers KILL_HARD_CAP.

    Construct a slow-creep curve that would CONTINUE except at hard cap.
    """
    # Linear creep over 80 sessions from 0 to ~16. Slope ≈ 0.2 (above flat),
    # but recent mean (last 10) ≈ 14, below criterion mean 24.
    scores = [i * 0.2 for i in range(80)]
    r = decide(scores)
    assert r.decision is Decision.KILL_HARD_CAP


def test_just_at_warmup_no_slope_check():
    """At exactly WARMUP+SLOPE_WINDOW-1, slope check should NOT fire yet."""
    # 19 sessions of zeros — past WARMUP=10 but not past WARMUP+SLOPE_WINDOW=20
    scores = [0] * 19
    r = decide(scores)
    # DEAD check fires before slope check; expected to be KILL_DEAD
    assert r.decision is Decision.KILL_DEAD


def test_just_at_warmup_no_dead_no_slope():
    """At exactly WARMUP, with non-zero recent activity, returns CONTINUE."""
    # 10 sessions, last 8 not all zero — dead won't fire, n < warmup+slope_window
    scores = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    r = decide(scores)
    assert r.decision is Decision.CONTINUE


# ---------------------------------------------------------------------------
# KillSwitchConfig overrides
# ---------------------------------------------------------------------------

def test_config_override_warmup():
    """Custom warmup lets us test on shorter histories."""
    cfg = KillSwitchConfig(warmup=3, slope_window=3, dead_window=3)
    r = decide([0, 0, 0, 0], cfg)
    assert r.decision is Decision.KILL_DEAD


def test_config_override_criterion_mean():
    cfg = KillSwitchConfig(criterion_mean=10.0)
    r = decide([12] * 15, cfg)
    assert r.decision is Decision.CRITERION_MET


# ---------------------------------------------------------------------------
# killed_at_payload schema
# ---------------------------------------------------------------------------

def test_killed_at_payload_has_required_keys():
    scores = [0] * 12
    r = decide(scores)
    payload = killed_at_payload(r, scores)
    assert payload["session"] == 12
    assert payload["decision"] == "kill_dead"
    assert "reason" in payload
    assert payload["scores_at_kill"] == list(scores)


# ---------------------------------------------------------------------------
# Decision flag helpers
# ---------------------------------------------------------------------------

def test_decision_flags():
    assert Decision.CONTINUE.is_kill is False
    assert Decision.CONTINUE.is_terminal is False
    assert Decision.KILL_DEAD.is_kill is True
    assert Decision.KILL_DEAD.is_terminal is True
    assert Decision.CRITERION_MET.is_kill is False
    assert Decision.CRITERION_MET.is_terminal is True
