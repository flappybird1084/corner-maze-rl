"""Validate that yoked action sequences form contiguous paths in grid space.

Checks that:
- FORWARD actions result in adjacent (manhattan=1) position changes
- TURN/PAUSE actions don't change position
- PICKUP actions move from corner to well (manhattan=1 diagonal OK)
- Direction updates are consistent with the action taken

Usage:
    python yoking/check_contiguity.py data/yoked/CM024_1e.parquet
    python yoking/check_contiguity.py --subject CM024
    python yoking/check_contiguity.py --all
"""
import argparse
import os
from glob import glob

import pandas as pd

# Action constants
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_PAUSE = 4

WELL_POSITIONS = {(1, 1), (11, 1), (1, 11), (11, 11)}
CORNER_TO_WELL = {(2, 2): (1, 1), (10, 2): (11, 1), (2, 10): (1, 11), (10, 10): (11, 11)}

DIR_TO_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
ACT_NAMES = {0: 'L', 1: 'R', 2: 'F', 3: 'PICKUP', 4: 'PAUSE'}
DIR_NAMES = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}


def check_contiguity(parquet_path, verbose=False):
    """Validate contiguity of a yoked parquet file.

    Returns list of (step, issue_description) tuples for each violation found.
    """
    df = pd.read_parquet(parquet_path)
    actions = df['action'].values
    gxs = df['grid_x'].values
    gys = df['grid_y'].values
    dirs = df['direction'].values

    issues = []

    for i in range(len(df) - 1):
        act = int(actions[i])
        pos = (int(gxs[i]), int(gys[i]))
        d = int(dirs[i])
        next_pos = (int(gxs[i + 1]), int(gys[i + 1]))
        next_dir = int(dirs[i + 1])

        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        manhattan = abs(dx) + abs(dy)

        if act == ACT_FORWARD:
            # After FORWARD: position should change by exactly 1 in the
            # facing direction, OR stay the same (blocked by wall/barrier).
            # Exception: FORWARD from inside a well exits diagonally to
            # the corner (manhattan=2) — this is expected.
            if pos in WELL_POSITIONS:
                # Well exit: allow manhattan <= 2
                if manhattan > 2:
                    issues.append((i, f'FORWARD from well {pos}: '
                                   f'jumped to {next_pos} (manhattan={manhattan})'))
            else:
                expected_delta = DIR_TO_DELTA[d]
                expected_next = (pos[0] + expected_delta[0], pos[1] + expected_delta[1])
                if next_pos != expected_next and next_pos != pos:
                    issues.append((i, f'FORWARD from {pos} d={DIR_NAMES[d]}: '
                                   f'expected {expected_next} or {pos} (blocked), '
                                   f'got {next_pos} (manhattan={manhattan})'))

        elif act == ACT_LEFT:
            # After LEFT: position may change (WELL_EXIT special case)
            # or stay the same (normal turn). Direction should change.
            expected_dir = (d - 1) % 4
            if next_pos != pos and manhattan > 1:
                # WELL_EXIT LEFT moves to adjacent corridor — manhattan 1 is OK
                issues.append((i, f'LEFT from {pos} d={DIR_NAMES[d]}: '
                               f'position jumped to {next_pos} (manhattan={manhattan})'))
            if next_dir != expected_dir and next_pos == pos:
                # Normal turn: direction should decrement by 1
                issues.append((i, f'LEFT from {pos}: dir {DIR_NAMES[d]}->{DIR_NAMES[next_dir]}, '
                               f'expected {DIR_NAMES[expected_dir]}'))

        elif act == ACT_RIGHT:
            # After RIGHT: position should not change.
            expected_dir = (d + 1) % 4
            if next_pos != pos:
                issues.append((i, f'RIGHT from {pos}: position changed to {next_pos}'))
            if next_dir != expected_dir:
                issues.append((i, f'RIGHT from {pos}: dir {DIR_NAMES[d]}->{DIR_NAMES[next_dir]}, '
                               f'expected {DIR_NAMES[expected_dir]}'))

        elif act == ACT_PICKUP:
            # After PICKUP: position moves from corner to well (OK) or
            # stays at well (turnaround inside). Allow manhattan <= 2.
            if manhattan > 2 and pos not in WELL_POSITIONS:
                issues.append((i, f'PICKUP from {pos}: jumped to {next_pos} '
                               f'(manhattan={manhattan})'))

        elif act == ACT_PAUSE:
            # After PAUSE: position and direction should not change.
            if next_pos != pos:
                issues.append((i, f'PAUSE from {pos}: position changed to {next_pos}'))
            if next_dir != d:
                issues.append((i, f'PAUSE from {pos}: dir changed '
                               f'{DIR_NAMES[d]}->{DIR_NAMES[next_dir]}'))

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Validate contiguity of yoked action sequences.',
    )
    parser.add_argument('parquet', nargs='?', type=str, default=None,
                        help='Path to a single parquet file.')
    parser.add_argument('--subject', type=str, default=None,
                        help='Check all sessions for a subject.')
    parser.add_argument('--all', action='store_true',
                        help='Check all parquet files in data/yoked/.')
    parser.add_argument('--dir', type=str, default='data/yoked',
                        help='Directory containing parquet files.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all issues, not just the first per file.')
    args = parser.parse_args()

    if args.parquet:
        files = [args.parquet]
    elif args.all or args.subject:
        pattern = os.path.join(args.dir, '*.parquet')
        files = sorted(glob(pattern))
        if args.subject:
            files = [f for f in files
                     if os.path.basename(f).startswith(args.subject + '_')]
    else:
        parser.error('Specify a parquet file, --subject, or --all')

    n_ok = 0
    n_fail = 0

    for fpath in files:
        name = os.path.basename(fpath).replace('.parquet', '')
        issues = check_contiguity(fpath, verbose=args.verbose)

        if not issues:
            n_ok += 1
            print(f'  OK  {name}')
        else:
            n_fail += 1
            print(f'  FAIL {name}: {len(issues)} issue(s)')
            if args.verbose:
                for step, desc in issues:
                    print(f'    step {step}: {desc}')
            else:
                step, desc = issues[0]
                print(f'    first: step {step}: {desc}')

    if len(files) > 1:
        print(f'\n{n_ok} ok, {n_fail} failed out of {len(files)} sessions')


if __name__ == '__main__':
    main()
