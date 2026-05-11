"""Validate that yoked action sequences form contiguous paths in grid space.

Checks that:
- FORWARD actions result in adjacent (manhattan=1) position changes
- TURN/PAUSE actions don't change position
- PICKUP actions move from corner to well (manhattan=1 diagonal OK)
- Direction updates are consistent with the action taken

Reads from the consolidated yoked dataset at ``data/yoked/dataset/``.

Usage:
    python -m corner_maze_rl.yoking.diagnostics.check_contiguity --subject CM024 --session 1e
    python -m corner_maze_rl.yoking.diagnostics.check_contiguity --subject CM024
    python -m corner_maze_rl.yoking.diagnostics.check_contiguity --all
    python -m corner_maze_rl.yoking.diagnostics.check_contiguity --all --variant real
"""
import argparse

import duckdb
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

# Variant → action-table filename
_ACTION_TABLE = {
    'synthetic': 'actions_synthetic_pretrial.parquet',
    'real':      'actions_real_pretrial.parquet',
    'exposure':  'actions_exposure.parquet',
}


def _variant_for_session_number(session_number: str) -> str:
    return 'exposure' if str(session_number).endswith('e') else 'synthetic'


def list_sessions(dataset_dir, subject=None, session=None, variant='auto'):
    """Return [(session_id, subject_name, session_number, variant), ...].

    ``variant='auto'``: synthetic for Acquisition, exposure for Exposure.
    ``variant in ('synthetic','real','exposure')``: restrict to that variant.
    ``variant='all'``: synthetic + real + exposure (one row per (sid, variant)).
    """
    where = ['1=1']
    if subject:
        where.append(f"sub.subject_name = '{subject}'")
    if session is not None:
        where.append(f"s.session_number = '{session}'")
    where_sql = ' AND '.join(where)
    rows = duckdb.sql(f"""
        SELECT s.session_id, sub.subject_name, s.session_number, s.session_phase
        FROM '{dataset_dir}/sessions.parquet' s
        JOIN '{dataset_dir}/subjects.parquet' sub USING (subject_id)
        WHERE {where_sql}
        ORDER BY sub.subject_name, s.session_number
    """).fetchall()

    out = []
    for sid, sub_name, sess_num, phase in rows:
        is_exp = (phase == 'Exposure')
        if variant == 'all':
            if is_exp:
                out.append((sid, sub_name, sess_num, 'exposure'))
            else:
                out.append((sid, sub_name, sess_num, 'synthetic'))
                out.append((sid, sub_name, sess_num, 'real'))
        elif variant == 'auto':
            out.append((sid, sub_name, sess_num, 'exposure' if is_exp else 'synthetic'))
        elif variant == 'exposure':
            if is_exp:
                out.append((sid, sub_name, sess_num, 'exposure'))
        else:  # 'synthetic' or 'real'
            if not is_exp:
                out.append((sid, sub_name, sess_num, variant))
    return out


def load_session(session_id, variant, dataset_dir):
    """Load action rows for one session from the consolidated dataset."""
    table = _ACTION_TABLE[variant]
    return duckdb.sql(f"""
        SELECT step, action, grid_x, grid_y, direction, rewarded
        FROM '{dataset_dir}/{table}'
        WHERE session_id = {session_id}
        ORDER BY step
    """).fetchdf()


def check_contiguity(df: pd.DataFrame, verbose: bool = False) -> list[tuple[int, str]]:
    """Validate contiguity of one session's action DataFrame.

    Returns list of (step, issue_description) tuples for each violation found.
    """
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
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject name (e.g., CM024). Required unless --all.')
    parser.add_argument('--session', type=str, default=None,
                        help='Session number (e.g., 1, 2e). Optional; if omitted, all sessions for the subject.')
    parser.add_argument('--variant', type=str, default='auto',
                        choices=['auto', 'synthetic', 'real', 'exposure', 'all'],
                        help="Action-table variant. 'auto' (default): synthetic for "
                             "Acquisition, exposure for Exposure. 'all': both pretrial variants for Acquisition.")
    parser.add_argument('--all', action='store_true',
                        help='Check every session in the dataset.')
    parser.add_argument('--dataset-dir', type=str, default='data/yoked/dataset',
                        help='Consolidated dataset directory.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all issues, not just the first per session.')
    args = parser.parse_args()

    if not args.all and args.subject is None:
        parser.error('Specify --subject (and optionally --session) or --all')

    sessions = list_sessions(
        args.dataset_dir,
        subject=args.subject if not args.all or args.subject else None,
        session=args.session,
        variant=args.variant,
    )
    if not sessions:
        print('No matching sessions found.')
        return

    n_ok = 0
    n_fail = 0

    for sid, sub_name, sess_num, variant in sessions:
        df = load_session(sid, variant, args.dataset_dir)
        label = f"{sub_name} {sess_num} ({variant})"
        issues = check_contiguity(df, verbose=args.verbose)

        if not issues:
            n_ok += 1
            print(f'  OK  {label}')
        else:
            n_fail += 1
            print(f'  FAIL {label}: {len(issues)} issue(s)')
            if args.verbose:
                for step, desc in issues:
                    print(f'    step {step}: {desc}')
            else:
                step, desc = issues[0]
                print(f'    first: step {step}: {desc}')

    if len(sessions) > 1:
        print(f'\n{n_ok} ok, {n_fail} failed out of {len(sessions)} sessions')


if __name__ == '__main__':
    main()
