"""Validation and shuffle utilities for trial sequence generation.

These functions were extracted from CornerMazeEnv.gen_grid_configuration_sequence()
to eliminate duplication across the 13 inner generator functions.
"""

import random

from .constants import THREEPEAT_MIN_SPACING


def validate_sequence_start_only(sequence, threepeat_limit, min_spacing=THREEPEAT_MIN_SPACING):
    """Check that start_arm (index 0) has no fourpeats, threepeat count
    within limit, and threepeats spaced at least min_spacing apart.

    Returns True if the sequence passes all checks.
    """
    threepeat_count = 0
    fourpeat_count = 0
    repeat_locs = []

    seq_len = len(sequence)
    for i, sgp in enumerate(sequence[:-3]):
        if sgp[0] == sequence[i + 1][0] and sgp[0] == sequence[i + 2][0] and sgp[0] == sequence[i + 3][0]:
            fourpeat_count += 1
        if sgp[0] == sequence[i + 1][0] and sgp[0] == sequence[i + 2][0]:
            threepeat_count += 1
            repeat_locs.append(i)
        if i == seq_len - 4 and sequence[i + 1][0] == sequence[i + 2][0] and sequence[i + 1][0] == sequence[i + 3][0]:
            threepeat_count += 1
            repeat_locs.append(i)

    if fourpeat_count > 0 or threepeat_count > threepeat_limit:
        return False

    # Check spacing between threepeats
    if len(repeat_locs) >= 2 and (repeat_locs[1] - repeat_locs[0]) < min_spacing:
        return False
    if len(repeat_locs) == 3 and (repeat_locs[2] - repeat_locs[1]) < min_spacing:
        return False

    return True


def validate_sequence_multi(sequence, start_limit=3, route_limit=3, goal_limit=3, min_spacing=THREEPEAT_MIN_SPACING):
    """Check start (idx 0), goal/cue (idx 1), and route (idx 2) for
    fourpeats, threepeat limits, and threepeat spacing.

    Returns True if the sequence passes all checks.
    """
    start_fourpeat = 0
    goal_fourpeat = 0
    route_fourpeat = 0
    start_threepeat = 0
    goal_threepeat = 0
    route_threepeat = 0
    start_repeat_loc = []
    goal_repeat_loc = []
    route_repeat_loc = []

    for i, sgp in enumerate(sequence[:-3]):
        if sgp[2] == sequence[i + 1][2] and sgp[2] == sequence[i + 2][2] and sgp[2] == sequence[i + 3][2]:
            route_fourpeat += 1
        if sgp[1] == sequence[i + 1][1] and sgp[1] == sequence[i + 2][1] and sgp[1] == sequence[i + 3][1]:
            goal_fourpeat += 1
        if sgp[0] == sequence[i + 1][0] and sgp[0] == sequence[i + 2][0] and sgp[0] == sequence[i + 3][0]:
            start_fourpeat += 1
        if sgp[2] == sequence[i + 1][2] and sgp[2] == sequence[i + 2][2]:
            route_threepeat += 1
            route_repeat_loc.append(i)
        if sgp[1] == sequence[i + 1][1] and sgp[1] == sequence[i + 2][1]:
            goal_threepeat += 1
            goal_repeat_loc.append(i)
        if sgp[0] == sequence[i + 1][0] and sgp[0] == sequence[i + 2][0]:
            start_threepeat += 1
            start_repeat_loc.append(i)

    # Check fourpeats and threepeat limits
    if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
            route_threepeat > route_limit or
            goal_threepeat > goal_limit or
            start_threepeat > start_limit):
        return False

    # Check spacing for each dimension
    for repeat_loc in [route_repeat_loc, goal_repeat_loc, start_repeat_loc]:
        if len(repeat_loc) >= 2 and (repeat_loc[1] - repeat_loc[0]) < min_spacing:
            return False
        if len(repeat_loc) == 3 and (repeat_loc[2] - repeat_loc[1]) < min_spacing:
            return False

    return True


def shuffle_uniform_chunks(base_pairs, chunk_size):
    """Shuffle base_pairs in place and concatenate chunk_size copies."""
    result = []
    for _ in range(chunk_size):
        random.shuffle(base_pairs)
        result += base_pairs.copy()
    return result


def shuffle_acq_then_probe(trained, probe, chunk_size, acq_chunks=2):
    """First acq_chunks shuffled from trained, remaining from probe."""
    result = []
    for i in range(chunk_size):
        if i <= acq_chunks - 1:
            random.shuffle(trained)
            result += trained.copy()
        else:
            random.shuffle(probe)
            result += probe.copy()
    return result


def shuffle_acq_then_novel(trained, probe, chunk_size, acq_chunks=2):
    """First acq_chunks from trained, then novel with forced-first item and dupe check.

    Returns (result_list, True) on success, ([], False) if dupe check failed.
    The probe list's last item is forced first in the chunk immediately after
    acquisition to ensure the first post-acquisition trial is a probe.
    """
    result = []
    temp_item = probe[-1]
    probe_len = len(probe)

    for i in range(chunk_size):
        if i <= acq_chunks - 1:
            random.shuffle(trained)
            result += trained.copy()
        elif i == acq_chunks:
            probe.remove(temp_item)
            random.shuffle(probe)
            result += [temp_item] + probe.copy()
            probe.append(temp_item)
        else:
            random.shuffle(probe)
            if result[-probe_len:] == probe:
                return [], False
            else:
                result += probe.copy()
    return result, True
