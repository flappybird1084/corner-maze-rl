"""Orientation-to-config mapping functions for trial sequence generation.

Each function maps an agent_cue_goal_orientation ('N/NE', 'N/SE', 'N/SW', 'N/NW')
and goal_location_index (0-3) to a list of (start_arm, cue, route, goal, tag) tuples.

These were extracted from CornerMazeEnv.gen_grid_configuration_sequence()
to eliminate duplicated orientation mapping blocks.

Terminology:
- gli: goal_location_index (0=NE, 1=SE, 2=SW, 3=NW)
- Routes: LL=0, RR=1, RL=2, LR=3
- f2: fixed second turn (alternating first turn)
- f1: fixed first turn (alternating second turn)
- Tags: 'trained', 'probe_trained', 'novel', 'reversal'
"""

from .constants import NUM_ARMS, NO_CUE

T = 'trained'
PT = 'probe_trained'
N = 'novel'
R = 'reversal'


# ============================================================
# f2 family: fixed second turn
# ============================================================

def get_f2_trained_pairs(orientation, gli):
    """Base trained pairs for f2 sessions (2 tuples).
    Used by: f2_acq, f2_novel_route (trained), f2_reversal (trained).
    """
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, gli, T),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, T)]
    elif orientation == 'N/SE':
        return [(gli, (gli - 1) % NUM_ARMS, 0, gli, T),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, gli, T)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 3, gli, T),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, T)]
    elif orientation == 'N/NW':
        return [((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, T),
                (gli, (gli + 1) % NUM_ARMS, 0, gli, T)]


def get_f2_no_cue_pairs(orientation, gli):
    """Same start/route/goal as f2 trained, but cue=4 (no cue).
    Used by: f2_no_cue, pi_novel_route_no_cue (trained), pi_reversal_no_cue (trained).
    """
    pairs = get_f2_trained_pairs(orientation, gli)
    return [(s, NO_CUE, r, g, tag) for s, c, r, g, tag in pairs]


def get_f2_novel_route_probe_pairs(orientation, gli):
    """Novel route probe pairs for f2 sessions (6 tuples: 2 probe_trained + 4 novel).
    Used by: f2_novel_route (probe portion).
    """
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, gli, PT),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, PT),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, N),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, N),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, N),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, N)]
    elif orientation == 'N/SE':
        return [(gli, (gli - 1) % NUM_ARMS, 0, gli, PT),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, gli, PT),
                ((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, N)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 3, gli, PT),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, PT),
                (gli, (gli - 2) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 2) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 2) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 2) % NUM_ARMS, 0, gli, N)]
    elif orientation == 'N/NW':
        return [((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, PT),
                (gli, (gli + 1) % NUM_ARMS, 0, gli, PT),
                ((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, N),
                ((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, N),
                ((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, N),
                ((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, N)]


def get_f2_reversal_probe_pairs(orientation, gli):
    """Reversal probe pairs for f2 sessions (2 tuples, goal shifted by +2).
    Used by: f2_reversal (probe portion).
    """
    rev_goal = (gli + 2) % NUM_ARMS
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, rev_goal, R),
                ((gli - 1) % NUM_ARMS, gli, 3, rev_goal, R)]
    elif orientation == 'N/SE':
        return [(gli, (gli - 1) % NUM_ARMS, 0, rev_goal, R),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, rev_goal, R)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 3, rev_goal, R),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, rev_goal, R)]
    elif orientation == 'N/NW':
        return [((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, rev_goal, R),
                (gli, (gli + 1) % NUM_ARMS, 0, rev_goal, R)]


def get_f2_rotate_pairs(orientation):
    """Rotate pairs for f2 sessions (8 tuples via range(NUM_ARMS)).
    Used by: f2_rotate, vc_acquisition, vc_novel_route_rotate (trained),
             vc_reversal_rotate (trained).
    """
    if orientation == 'N/NE':
        return [((i + 1) % NUM_ARMS, i, 1, i, T) for i in range(NUM_ARMS)] + [((i - 1) % NUM_ARMS, i, 3, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/SE':
        return [(i, (i - 1) % NUM_ARMS, 0, i, T) for i in range(NUM_ARMS)] + [((i + 2) % NUM_ARMS, (i - 1) % NUM_ARMS, 2, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/SW':
        return [((i + 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 1, i, T) for i in range(NUM_ARMS)] + [((i - 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 3, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/NW':
        return [(i, (i + 1) % NUM_ARMS, 0, i, T) for i in range(NUM_ARMS)] + [((i + 2) % NUM_ARMS, (i + 1) % NUM_ARMS, 2, i, T) for i in range(NUM_ARMS)]


# ============================================================
# f1 family: fixed first turn
# ============================================================

def get_f1_trained_pairs(orientation, gli):
    """Base trained pairs for f1 sessions (2 tuples).
    Used by: f1_acq, f1_novel_route (trained), f1_reversal (trained).
    """
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, gli, T),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, T)]
    elif orientation == 'N/SE':
        return [((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, T),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, gli, T)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli + 2) % NUM_ARMS, 3, gli, T),
                (gli, (gli + 2) % NUM_ARMS, 0, gli, T)]
    elif orientation == 'N/NW':
        return [((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, T),
                (gli, (gli + 1) % NUM_ARMS, 0, gli, T)]


def get_f1_no_cue_pairs(orientation, gli):
    """Same start/route/goal as f1 trained, but cue=4 (no cue).
    Used by: f1_no_cue.
    """
    pairs = get_f1_trained_pairs(orientation, gli)
    return [(s, NO_CUE, r, g, tag) for s, c, r, g, tag in pairs]


def get_f1_novel_route_probe_pairs(orientation, gli):
    """Novel route probe pairs for f1 sessions (6 tuples: 2 probe_trained + 4 novel).
    Used by: f1_novel_route (probe portion).
    """
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, gli, PT),
                ((gli + 2) % NUM_ARMS, gli, 2, gli, PT),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, N),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, N),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, N),
                ((gli - 1) % NUM_ARMS, gli, 3, gli, N)]
    elif orientation == 'N/SE':
        return [((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, gli, PT),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, gli, PT),
                (gli, (gli - 1) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 1) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 1) % NUM_ARMS, 0, gli, N),
                (gli, (gli - 1) % NUM_ARMS, 0, gli, N)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 3, gli, PT),
                (gli, (gli - 2) % NUM_ARMS, 0, gli, PT),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, N),
                ((gli + 1) % NUM_ARMS, (gli - 2) % NUM_ARMS, 1, gli, N)]
    elif orientation == 'N/NW':
        return [((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, gli, PT),
                (gli, (gli + 1) % NUM_ARMS, 0, gli, PT),
                ((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, N),
                ((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, N),
                ((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, N),
                ((gli + 2) % NUM_ARMS, (gli + 1) % NUM_ARMS, 2, gli, N)]


def get_f1_reversal_probe_pairs(orientation, gli):
    """Reversal probe pairs for f1 sessions (2 tuples, goal shifted by +2).
    Used by: f1_reversal (probe portion).
    """
    rev_goal = (gli + 2) % NUM_ARMS
    if orientation == 'N/NE':
        return [((gli + 1) % NUM_ARMS, gli, 1, rev_goal, R),
                ((gli + 2) % NUM_ARMS, gli, 2, rev_goal, R)]
    elif orientation == 'N/SE':
        return [((gli + 1) % NUM_ARMS, (gli - 1) % NUM_ARMS, 1, rev_goal, R),
                ((gli + 2) % NUM_ARMS, (gli - 1) % NUM_ARMS, 2, rev_goal, R)]
    elif orientation == 'N/SW':
        return [((gli - 1) % NUM_ARMS, (gli + 2) % NUM_ARMS, 3, rev_goal, R),
                (gli, (gli + 2) % NUM_ARMS, 0, rev_goal, R)]
    elif orientation == 'N/NW':
        return [((gli - 1) % NUM_ARMS, (gli + 1) % NUM_ARMS, 3, rev_goal, R),
                (gli, (gli + 1) % NUM_ARMS, 0, rev_goal, R)]


def get_f1_rotate_pairs(orientation):
    """Rotate pairs for f1 sessions (8 tuples via range(NUM_ARMS)).
    Used by: f1_rotate.
    """
    if orientation == 'N/NE':
        return [((i + 1) % NUM_ARMS, i, 1, i, T) for i in range(NUM_ARMS)] + [((i + 2) % NUM_ARMS, i, 2, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/SE':
        return [((i + 1) % NUM_ARMS, (i - 1) % NUM_ARMS, 1, i, T) for i in range(NUM_ARMS)] + [((i + 2) % NUM_ARMS, (i - 1) % NUM_ARMS, 2, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/SW':
        return [(i, (i + 2) % NUM_ARMS, 0, i, T) for i in range(NUM_ARMS)] + [((i - 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 3, i, T) for i in range(NUM_ARMS)]
    elif orientation == 'N/NW':
        return [(i, (i + 1) % NUM_ARMS, 0, i, T) for i in range(NUM_ARMS)] + [((i - 1) % NUM_ARMS, (i + 1) % NUM_ARMS, 3, i, T) for i in range(NUM_ARMS)]


# ============================================================
# PI-only family (no cue versions of f2)
# ============================================================

def get_pi_novel_route_no_cue_probe_pairs(orientation, gli):
    """Novel route probe pairs for PI no-cue sessions (6 tuples, cue=4).
    Used by: pi_novel_route_no_cue (probe portion).
    Same formulas as f2 novel route probe but with cue=4.
    """
    pairs = get_f2_novel_route_probe_pairs(orientation, gli)
    return [(s, NO_CUE, r, g, tag) for s, c, r, g, tag in pairs]


def get_pi_reversal_no_cue_probe_pairs(orientation, gli):
    """Reversal probe pairs for PI no-cue sessions (2 tuples, cue=4, goal shifted).
    Used by: pi_reversal_no_cue (probe portion).
    Same formulas as f2 reversal probe but with cue=4.
    """
    pairs = get_f2_reversal_probe_pairs(orientation, gli)
    return [(s, NO_CUE, r, g, tag) for s, c, r, g, tag in pairs]


# ============================================================
# VC family (visual cue only, rotate-based)
# ============================================================

def get_vc_novel_route_rotate_probe_pairs(orientation):
    """Novel route probe pairs for VC rotate sessions (24 tuples: 8 probe_trained + 16 novel).
    Used by: vc_novel_route_rotate (probe portion).
    """
    if orientation == 'N/NE':
        return [((i + 1) % NUM_ARMS, i, 1, i, PT) for i in range(NUM_ARMS)] + \
               [((i - 1) % NUM_ARMS, i, 3, i, PT) for i in range(NUM_ARMS)] + \
               [((i + 2) % NUM_ARMS, i % NUM_ARMS, 2, i % NUM_ARMS, N) for i in range(16)]
    elif orientation == 'N/SE':
        return [(i, (i - 1) % NUM_ARMS, 0, i, PT) for i in range(NUM_ARMS)] + \
               [((i + 2) % NUM_ARMS, (i - 1) % NUM_ARMS, 2, i, PT) for i in range(NUM_ARMS)] + \
               [((i + 1) % NUM_ARMS, (i - 1) % NUM_ARMS, 1, i % NUM_ARMS, N) for i in range(16)]
    elif orientation == 'N/SW':
        return [((i + 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 1, i, PT) for i in range(NUM_ARMS)] + \
               [((i - 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 3, i, PT) for i in range(NUM_ARMS)] + \
               [(i % NUM_ARMS, (i + 2) % NUM_ARMS, 0, i % NUM_ARMS, N) for i in range(16)]
    elif orientation == 'N/NW':
        return [(i, (i + 1) % NUM_ARMS, 0, i, PT) for i in range(NUM_ARMS)] + \
               [((i + 2) % NUM_ARMS, (i + 1) % NUM_ARMS, 2, i, PT) for i in range(NUM_ARMS)] + \
               [((i - 1) % NUM_ARMS, (i + 1) % NUM_ARMS, 3, i % NUM_ARMS, N) for i in range(16)]


def get_vc_reversal_rotate_probe_pairs(orientation):
    """Reversal probe pairs for VC rotate sessions (8 tuples, goal shifted by +2).
    Used by: vc_reversal_rotate (probe portion).
    """
    if orientation == 'N/NE':
        return [((i + 1) % NUM_ARMS, i, 1, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)] + \
               [((i - 1) % NUM_ARMS, i, 3, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)]
    elif orientation == 'N/SE':
        return [(i, (i - 1) % NUM_ARMS, 0, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)] + \
               [((i + 2) % NUM_ARMS, (i - 1) % NUM_ARMS, 2, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)]
    elif orientation == 'N/SW':
        return [((i + 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 1, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)] + \
               [((i - 1) % NUM_ARMS, (i + 2) % NUM_ARMS, 3, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)]
    elif orientation == 'N/NW':
        return [(i, (i + 1) % NUM_ARMS, 0, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)] + \
               [((i + 2) % NUM_ARMS, (i + 1) % NUM_ARMS, 2, (i + 2) % NUM_ARMS, R) for i in range(NUM_ARMS)]
