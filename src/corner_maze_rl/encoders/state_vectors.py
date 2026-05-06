"""Tabular state-vector helpers for PPO and SR.

Ported verbatim from legacy ``src/rl/custom_rl.py`` (top of file). These
functions back the tabular PPO and SR comparison agents per plan §16.3.
The plan asks to split into ``one_hot_pose.py`` + ``reward_history.py``
encoders, but the legacy implementation packs both halves into single
helpers; splitting is cosmetic and would just dilute test coverage. The
helpers are exposed here, and thin StateEncoder wrappers can be added
later if compositional access (per plan §5.3) becomes needed.

Output dimensions:
  * ``generate_state_vector``        — 49*4 + 4*n_wm = 236 (default n_wm=10)
  * ``generate_state_vector_onehot`` — 236, no adjacent-cell activation
  * ``generate_state_vector_phase``  — 236 + 3 (phase one-hot) = 239
"""
from __future__ import annotations

import numpy as np

from corner_maze_rl.env.constants import WELL_LOCATIONS

# All valid floor positions in the 13x13 maze (49 locations).
_MAZE_FLOOR: set[tuple[int, int]] = set()
for _i in range(3):
    for _j in range(9):
        _MAZE_FLOOR.add((_j + 2, 4 * _i + 2))
for _i in range(3):
    for _j in range(3):
        _MAZE_FLOOR.add((4 * _i + 2, _j + 3))
        _MAZE_FLOOR.add((4 * _i + 2, _j + 7))
for _wp in WELL_LOCATIONS:
    _MAZE_FLOOR.add(tuple(_wp))

VALID_LOCATIONS: list[tuple[int, int]] = sorted(_MAZE_FLOOR)
LOCATION_TO_INDEX: dict[tuple[int, int], int] = {
    pos: i for i, pos in enumerate(VALID_LOCATIONS)
}
NUM_LOCATIONS: int = len(VALID_LOCATIONS)  # 49


# ---------------------------------------------------------------------------
# Phase constants (matches legacy)
# ---------------------------------------------------------------------------

PHASE_TRIAL = 0
PHASE_PRETRIAL = 1
PHASE_ITI = 2
NUM_PHASES = 3

# env state_type → behavioural-phase one-hot index
_STATE_TYPE_TO_PHASE = {
    0: PHASE_ITI,       # STATE_BASE
    1: PHASE_ITI,       # STATE_EXPA
    2: PHASE_ITI,       # STATE_EXPB
    3: PHASE_PRETRIAL,
    4: PHASE_TRIAL,
    5: PHASE_ITI,
}


# ---------------------------------------------------------------------------
# Pose vectors
# ---------------------------------------------------------------------------

def generate_state_vector(pose, reward_timers, n_wm_units: int = 10, env=None) -> np.ndarray:
    """Pose with adjacent-cell activation + decaying reward timers.

    Pose part: 196-dim (4 directions × 49 cells), with current cell + 4-conn
    neighbours activated under the agent's current heading. WM part:
    4 wells × n_wm decaying activation.
    """
    x, y, d = pose

    pose_vec = np.zeros(NUM_LOCATIONS * 4)
    direction_offset = d * NUM_LOCATIONS
    if (x, y) in LOCATION_TO_INDEX:
        cur = LOCATION_TO_INDEX[(x, y)]
        pose_vec[direction_offset + cur] = 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (x + dx, y + dy)
            if nb in LOCATION_TO_INDEX:
                pose_vec[direction_offset + LOCATION_TO_INDEX[nb]] = 1

    wm_vec = _wm_vector(reward_timers, n_wm_units)
    return np.concatenate((pose_vec, wm_vec)).astype(np.float32)


def generate_state_vector_onehot(pose, reward_timers, n_wm_units: int = 10, env=None) -> np.ndarray:
    """Pure one-hot pose (no adjacency) + decaying reward timers."""
    x, y, d = pose

    pose_vec = np.zeros(NUM_LOCATIONS * 4)
    direction_offset = d * NUM_LOCATIONS
    if (x, y) in LOCATION_TO_INDEX:
        pose_vec[direction_offset + LOCATION_TO_INDEX[(x, y)]] = 1

    wm_vec = _wm_vector(reward_timers, n_wm_units)
    return np.concatenate((pose_vec, wm_vec)).astype(np.float32)


def generate_state_vector_phase(pose, reward_timers, n_wm_units: int = 10, env=None) -> np.ndarray:
    """Pose + phase one-hot + decaying reward timers (239-dim default).

    Reads the env's current ``state_type`` to set the phase one-hot. If
    ``env`` is None defaults to PHASE_TRIAL (4).
    """
    if env is not None:
        try:
            entry = env.grid_configuration_sequence[env.sequence_count]
            head = entry[0]
            state_type = int(head) if isinstance(head, (int, float)) else int(entry[0][0])
        except Exception:
            state_type = 4
    else:
        state_type = 4

    x, y, d = pose
    pose_vec = np.zeros(NUM_LOCATIONS * 4)
    direction_offset = d * NUM_LOCATIONS
    if (x, y) in LOCATION_TO_INDEX:
        pose_vec[direction_offset + LOCATION_TO_INDEX[(x, y)]] = 1

    phase_vec = np.zeros(NUM_PHASES)
    phase_vec[_STATE_TYPE_TO_PHASE.get(state_type, PHASE_ITI)] = 1

    wm_vec = _wm_vector(reward_timers, n_wm_units)
    return np.concatenate((pose_vec, phase_vec, wm_vec)).astype(np.float32)


def _wm_vector(reward_timers, n_wm_units: int) -> np.ndarray:
    wm = np.zeros(4 * n_wm_units)
    for corner_idx in range(4):
        time_since = reward_timers[corner_idx]
        active = max(0, n_wm_units - time_since)
        start = corner_idx * n_wm_units
        wm[start: start + active] = 1
    return wm


# ---------------------------------------------------------------------------
# Dimensionality + projection helpers
# ---------------------------------------------------------------------------

def compute_obs_dim(n_wm_units: int = 10) -> int:
    return NUM_LOCATIONS * 4 + 4 * n_wm_units  # 236 default


def compute_obs_dim_phase(n_wm_units: int = 10) -> int:
    return NUM_LOCATIONS * 4 + NUM_PHASES + 4 * n_wm_units  # 239


def build_position_projection_matrix(n_wm_units: int = 10) -> np.ndarray:
    """Project 236-dim full state to 89-dim position-only state.

    Sums over the 4 direction slices and passes WM through unchanged.
    Used by SR's position-only ``w`` mode.
    """
    obs_dim = NUM_LOCATIONS * 4 + 4 * n_wm_units
    pos_dim = NUM_LOCATIONS + 4 * n_wm_units
    P = np.zeros((obs_dim, pos_dim), dtype=np.float32)
    for d in range(4):
        for i in range(NUM_LOCATIONS):
            P[d * NUM_LOCATIONS + i, i] = 1.0
    wm_start_full = NUM_LOCATIONS * 4
    wm_start_pos = NUM_LOCATIONS
    for j in range(4 * n_wm_units):
        P[wm_start_full + j, wm_start_pos + j] = 1.0
    return P
