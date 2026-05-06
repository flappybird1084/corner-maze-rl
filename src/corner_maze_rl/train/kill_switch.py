"""Compute-aware early-termination for training runs.

The kill switch watches a per-session learning curve (one scalar per session,
typically `perfect_trial_count` ∈ [0, 32]) and decides whether the run is
worth continuing. See ``md/dt-repo-plan.md`` §8 for the full rationale.

Decision states (the four return values of ``decide``):

  * ``CONTINUE``     — keep training.
  * ``CRITERION_MET`` — recent mean ≥ ``CRITERION_MEAN``; positive early stop.
  * ``KILL_DEAD``    — no successful trial in last ``DEAD_WINDOW`` sessions.
  * ``KILL_FLAT``    — slope is below ``FLAT_SLOPE_EPS`` AND recent mean is
                      below ``ABSOLUTE_FLOOR``. The asymmetric ``<`` (not
                      ``abs(slope) <``) is intentional: declining curves get
                      killed too. Creeping-up curves with slope above
                      ``FLAT_SLOPE_EPS`` survive.
  * ``KILL_HARD_CAP`` — ``HARD_CAP`` sessions reached.

All thresholds are module-level constants, overridable via the
``KillSwitchConfig`` dataclass for per-run tuning (``profile`` flag in
``scripts/train.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Defaults — see plan §8 for rationale
# ---------------------------------------------------------------------------

WARMUP: int = 10              # never kill before this many sessions
SLOPE_WINDOW: int = 10        # regression window for "creeping-up" detection
FLAT_SLOPE_EPS: float = 0.05  # trials/session — below this is "not learning"
ABSOLUTE_FLOOR: float = 4.0   # mean of last window must be < this to kill
DEAD_WINDOW: int = 8          # if no successful trial in K sessions, kill
CRITERION_MEAN: float = 24.0  # 24/32 = 75% perfect → declared learned
HARD_CAP: int = 80            # never run past this


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    CONTINUE = "continue"
    CRITERION_MET = "criterion_met"
    KILL_DEAD = "kill_dead"
    KILL_FLAT = "kill_flat"
    KILL_HARD_CAP = "kill_hard_cap"

    @property
    def is_kill(self) -> bool:
        return self.name.startswith("KILL_")

    @property
    def is_terminal(self) -> bool:
        return self is not Decision.CONTINUE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KillSwitchConfig:
    """Per-run overrides. Defaults mirror the module-level constants."""
    warmup: int = WARMUP
    slope_window: int = SLOPE_WINDOW
    flat_slope_eps: float = FLAT_SLOPE_EPS
    absolute_floor: float = ABSOLUTE_FLOOR
    dead_window: int = DEAD_WINDOW
    criterion_mean: float = CRITERION_MEAN
    hard_cap: int = HARD_CAP


DEFAULT_CONFIG = KillSwitchConfig()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def linear_regression_slope(values: Sequence[float]) -> float:
    """Closed-form OLS slope for y = a + b*t with t = 0..N-1.

    Returns 0.0 when N < 2 (no slope defined). Pure Python; no numpy
    dependency in this hot-path helper, since the kill switch is called once
    per session — small N, want zero allocation overhead.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_t = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = 0.0
    den = 0.0
    for t, y in enumerate(values):
        dt = t - mean_t
        num += dt * (y - mean_y)
        den += dt * dt
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    reason: str
    slope: float | None = None
    recent_mean: float | None = None
    n_sessions: int = 0

    @property
    def should_stop(self) -> bool:
        return self.decision.is_terminal


def decide(
    scores: Sequence[float],
    config: KillSwitchConfig = DEFAULT_CONFIG,
) -> DecisionResult:
    """Evaluate the kill-switch state given the current per-session score history.

    Parameters
    ----------
    scores
        Full session-score history so far, e.g. ``[s_0, s_1, ..., s_{n-1}]``.
        Each entry is a scalar (typically perfect-trial count, 0..32).
    config
        Threshold overrides; defaults to module constants.

    Returns
    -------
    DecisionResult
        ``decision`` is one of the ``Decision`` values; auxiliary fields
        report what the decision was based on (slope, recent mean, n).
    """
    n = len(scores)

    if n < config.warmup:
        return DecisionResult(Decision.CONTINUE, "warmup", n_sessions=n)

    # Hard cap takes precedence over success — if you ran HARD_CAP sessions
    # and still need this check, something's wrong. But we still want to
    # report criterion_met on a healthy plateau before bailing.
    if n >= config.dead_window and all(
        s == 0 for s in scores[-config.dead_window:]
    ):
        return DecisionResult(
            Decision.KILL_DEAD,
            f"no successful trial in last {config.dead_window} sessions",
            n_sessions=n,
        )

    window = list(scores[-config.slope_window:])
    slope = linear_regression_slope(window)
    recent_mean = sum(window) / len(window)

    # Positive criterion: high mean → hand off to positive early stop.
    if recent_mean >= config.criterion_mean:
        return DecisionResult(
            Decision.CRITERION_MET,
            f"recent_mean={recent_mean:.1f} ≥ {config.criterion_mean}",
            slope=slope,
            recent_mean=recent_mean,
            n_sessions=n,
        )

    # Flat-or-declining + low-floor: kill.
    # Asymmetric `<`: a declining curve (slope < 0) is killed too, since
    # a learning model shouldn't be regressing. A creeping-up curve with
    # slope > FLAT_SLOPE_EPS survives even at a low recent mean.
    if (
        n >= config.warmup + config.slope_window
        and slope < config.flat_slope_eps
        and recent_mean < config.absolute_floor
    ):
        return DecisionResult(
            Decision.KILL_FLAT,
            f"slope={slope:.3f} < {config.flat_slope_eps}, "
            f"mean={recent_mean:.1f} < {config.absolute_floor}",
            slope=slope,
            recent_mean=recent_mean,
            n_sessions=n,
        )

    if n >= config.hard_cap:
        return DecisionResult(
            Decision.KILL_HARD_CAP,
            f"hard cap {config.hard_cap} reached",
            slope=slope,
            recent_mean=recent_mean,
            n_sessions=n,
        )

    return DecisionResult(
        Decision.CONTINUE,
        "still training",
        slope=slope,
        recent_mean=recent_mean,
        n_sessions=n,
    )


# ---------------------------------------------------------------------------
# Persistence — used by runner to log a kill event
# ---------------------------------------------------------------------------

def killed_at_payload(result: DecisionResult, scores: Sequence[float]) -> dict:
    """Build the JSON payload for ``runs/.../killed_at.json``.

    Mirrors the schema described in plan §7.1 / §8.5.
    """
    return {
        "session": result.n_sessions,
        "decision": result.decision.value,
        "reason": result.reason,
        "slope": result.slope,
        "recent_mean": result.recent_mean,
        "scores_at_kill": list(scores),
    }
