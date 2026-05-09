"""Validate that yoked action well visits match real rat tracking data.

Compares PICKUP actions (action=3) in the yoked dataset against the
trial_well_visits table from corner-maze-analysis for each acquisition
session.

Two comparison modes:
  --rewarded-only (default): Compare only rewarded well visits.
      These are the goal-well arrivals that end each trial. The real
      source is trial_well_visits (is_reward=true), which only covers
      the trial phase. This is the apples-to-apples comparison.
  --all-visits: Compare ALL well visits (rewarded + unrewarded).
      Note: trial_well_visits only records visits during the "trial"
      phase, while yoked PICKUP actions include pretrial and ITI well
      entries too. Expect mismatches from this phase mismatch.

Checks:
  1. Same number of well visits
  2. Same wells visited in the same order
  3. Same reward flags (--all-visits mode)

Exposure sessions are skipped (no trial_well_visits data).

Usage:
    python yoking/check_well_visits.py --all
    python yoking/check_well_visits.py --all --all-visits
    python yoking/check_well_visits.py --subject CM025
    python yoking/check_well_visits.py --subject CM025 --session 3 -v
"""
import argparse
import os

import duckdb

from corner_maze_rl.yoking.data_loader import ANALYSIS_DATA_DIR, _ZONE_TO_WELL_POS

# Repo root = src/corner_maze_rl/yoking/diagnostics/check_well_visits.py → up 4
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'data', 'yoked', 'dataset',
)

# Grid position → well name for readable output
_WELL_POS_TO_NAME = {(11, 1): 'NE', (11, 11): 'SE', (1, 11): 'SW', (1, 1): 'NW'}

# Corner positions → adjacent well positions
_CORNER_TO_WELL = {(2, 2): (1, 1), (10, 2): (11, 1),
                   (2, 10): (1, 11), (10, 10): (11, 11)}

ACT_PICKUP = 3


def _get_real_well_visits(session_id: int,
                          rewarded_only: bool = True,
                          ) -> list[tuple[tuple[int, int], bool]]:
    """Get well visits from real tracking data for an acquisition session.

    Returns list of (well_grid_pos, is_reward) in chronological order.
    """
    path = os.path.join(ANALYSIS_DATA_DIR, 'trial_well_visits.parquet')
    reward_filter = "AND is_reward = true" if rewarded_only else ""
    df = duckdb.sql(f"""
        SELECT well_zone, is_reward
        FROM '{path}'
        WHERE session_id = {session_id}
        {reward_filter}
        ORDER BY t_entry_ms
    """).fetchdf()

    visits = []
    for _, row in df.iterrows():
        well_pos = _ZONE_TO_WELL_POS.get(int(row['well_zone']))
        if well_pos is not None:
            visits.append((well_pos, bool(row['is_reward'])))
    return visits


def _get_yoked_well_visits(session_id: int,
                           rewarded_only: bool = True,
                           ) -> list[tuple[tuple[int, int], bool]]:
    """Get PICKUP actions from the yoked dataset for a session.

    The grid position recorded is the corner position BEFORE the pickup.
    Maps corner → adjacent well position.
    """
    actions_path = os.path.join(DATASET_DIR, 'actions_synthetic_pretrial.parquet')
    reward_filter = "AND rewarded = 1" if rewarded_only else ""
    df = duckdb.sql(f"""
        SELECT grid_x, grid_y, rewarded
        FROM '{actions_path}'
        WHERE session_id = {session_id}
          AND action = {ACT_PICKUP}
          {reward_filter}
        ORDER BY step
    """).fetchdf()

    well_positions = set(_CORNER_TO_WELL.values())

    visits = []
    for _, row in df.iterrows():
        pos = (int(row['grid_x']), int(row['grid_y']))
        if pos in _CORNER_TO_WELL:
            well_pos = _CORNER_TO_WELL[pos]
        elif pos in well_positions:
            well_pos = pos
        else:
            well_pos = pos
        visits.append((well_pos, bool(row['rewarded'])))
    return visits


def check_session(session_id: int, subject_name: str,
                  session_number: str, rewarded_only: bool = True,
                  verbose: bool = False) -> tuple[str, str]:
    """Compare well visits for a single session.

    Returns (status, detail) where status is 'OK', 'MISMATCH', or 'SKIP'.
    """
    real = _get_real_well_visits(session_id, rewarded_only=rewarded_only)
    yoked = _get_yoked_well_visits(session_id, rewarded_only=rewarded_only)

    if not real and not yoked:
        return ('SKIP', 'no well visits in either source')

    issues = []

    # Check count
    if len(real) != len(yoked):
        issues.append(f'count: real={len(real)} yoked={len(yoked)}')

    # Check well-by-well
    n = min(len(real), len(yoked))
    well_mismatches = 0
    reward_mismatches = 0
    first_mismatch = None

    for i in range(n):
        real_pos, real_rew = real[i]
        yoked_pos, yoked_rew = yoked[i]

        if real_pos != yoked_pos:
            well_mismatches += 1
            if first_mismatch is None:
                real_name = _WELL_POS_TO_NAME.get(real_pos, str(real_pos))
                yoked_name = _WELL_POS_TO_NAME.get(yoked_pos, str(yoked_pos))
                first_mismatch = f'visit {i}: real={real_name} yoked={yoked_name}'

        if real_rew != yoked_rew:
            reward_mismatches += 1
            if first_mismatch is None:
                well_name = _WELL_POS_TO_NAME.get(real_pos, str(real_pos))
                first_mismatch = (f'visit {i} ({well_name}): '
                                  f'real_rew={real_rew} yoked_rew={yoked_rew}')

    if well_mismatches:
        issues.append(f'{well_mismatches} well mismatch(es)')
    if reward_mismatches:
        issues.append(f'{reward_mismatches} reward mismatch(es)')

    if not issues:
        return ('OK', f'{len(real)} visits match')

    detail = '; '.join(issues)
    if first_mismatch:
        detail += f' | first: {first_mismatch}'

    if verbose:
        lines = []
        for i in range(max(len(real), len(yoked))):
            r = real[i] if i < len(real) else None
            y = yoked[i] if i < len(yoked) else None
            r_str = f'{_WELL_POS_TO_NAME.get(r[0], str(r[0]))}({"R" if r[1] else "-"})' if r else '---'
            y_str = f'{_WELL_POS_TO_NAME.get(y[0], str(y[0]))}({"R" if y[1] else "-"})' if y else '---'
            marker = ' *' if r != y else ''
            lines.append(f'    {i:3d}: real={r_str:10s} yoked={y_str:10s}{marker}')
        detail += '\n' + '\n'.join(lines)

    return ('MISMATCH', detail)


def main():
    parser = argparse.ArgumentParser(
        description='Validate yoked well visits against real tracking data.',
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Check all acquisition sessions for a subject.')
    parser.add_argument('--session', type=str, default=None,
                        help='Filter to a specific session number (use with --subject).')
    parser.add_argument('--all', action='store_true',
                        help='Check all acquisition sessions in the dataset.')
    parser.add_argument('--all-visits', action='store_true',
                        help='Compare all well visits, not just rewarded. '
                             'Note: expect mismatches due to ITI/pretrial '
                             'visits in yoked data not present in '
                             'trial_well_visits.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show full visit-by-visit comparison on mismatch.')
    args = parser.parse_args()

    if not args.subject and not args.all:
        parser.error('Specify --subject or --all')

    rewarded_only = not args.all_visits

    sessions_path = os.path.join(DATASET_DIR, 'sessions.parquet')
    subjects_path = os.path.join(DATASET_DIR, 'subjects.parquet')

    conditions = ["s.session_phase = 'Acquisition'"]
    if args.subject:
        conditions.append(f"sub.subject_name = '{args.subject}'")
    if args.session:
        conditions.append(f"s.session_number = '{args.session}'")

    where = ' AND '.join(conditions)
    sessions = duckdb.sql(f"""
        SELECT s.session_id, sub.subject_name, s.session_number
        FROM '{sessions_path}' s
        JOIN '{subjects_path}' sub USING (subject_id)
        WHERE {where}
        ORDER BY sub.subject_name, s.session_number
    """).fetchdf()

    if len(sessions) == 0:
        print('No matching acquisition sessions found.')
        return

    mode = 'all visits' if args.all_visits else 'rewarded only'
    print(f'Mode: {mode}\n')

    n_ok = 0
    n_fail = 0
    n_skip = 0
    failures = []

    for _, row in sessions.iterrows():
        session_id = int(row['session_id'])
        subject = row['subject_name']
        sess_num = row['session_number']
        name = f'{subject}_{sess_num}'

        try:
            status, detail = check_session(
                session_id, subject, sess_num,
                rewarded_only=rewarded_only, verbose=args.verbose)
            if status == 'OK':
                n_ok += 1
                print(f'  OK    {name}: {detail}')
            elif status == 'SKIP':
                n_skip += 1
                print(f'  SKIP  {name}: {detail}')
            else:
                n_fail += 1
                failures.append((name, detail))
                print(f'  FAIL  {name}: {detail}')
        except Exception as e:
            n_fail += 1
            failures.append((name, str(e)))
            print(f'  ERR   {name}: {e}')

    total = n_ok + n_fail + n_skip
    print(f'\n{n_ok} ok, {n_fail} failed, {n_skip} skipped out of {total} sessions')

    if failures:
        print(f'\nFailed sessions ({n_fail}):')
        for name, detail in failures:
            summary = detail.split('\n')[0]
            print(f'  {name}: {summary}')


if __name__ == '__main__':
    main()
