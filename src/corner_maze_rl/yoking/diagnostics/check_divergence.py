"""Headless divergence checker for yoked action data.

Replays each yoked session through ``CornerMazeEnv`` and compares the env
agent's ``(grid_x, grid_y, direction)`` against the recorded yoked tracking
at every step. Reports the first divergence point or ``OK`` for each session.

Two input modes:

  * **Per-session parquets** (legacy): expects files like
    ``data/yoked/CM016_5.parquet`` with file-level metadata, as written by
    ``build_yoked.py``. Use ``--dir <path>``.

  * **Normalized dataset** (current): reads from the 5-table dataset under
    ``data/yoked/dataset/``. Joins ``sessions.parquet`` + ``subjects.parquet``
    to look up ``training_group`` + ``cue_goal_orientation``, picks the env
    paradigm via ``PARADIGM_MAP``, and pulls actions from the appropriate
    ``actions_*.parquet`` based on ``session_phase``. Use ``--from-dataset``.

Usage:
    python -m corner_maze_rl.yoking.diagnostics.check_divergence --from-dataset
    python -m corner_maze_rl.yoking.diagnostics.check_divergence --from-dataset --subject CM057
    python -m corner_maze_rl.yoking.diagnostics.check_divergence --dir data/yoked
"""
import argparse
import json
import os
from glob import glob

import pandas as pd

from corner_maze_rl.data.session_types import map_session_to_env_kwargs
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv

from .replay_session import load_yoked_parquet, _inject_trial_configs

_GOAL_IDX_TO_NAME = {0: 'NE', 1: 'SE', 2: 'SW', 3: 'NW'}


def _run_replay(env, actions_df):
    """Step through actions, comparing env state to recorded yoked state.

    Returns (status, detail). ``status`` is 'OK' or 'DIVERGE'.
    """
    actions = actions_df['action'].values
    grid_xs = actions_df['grid_x'].values
    grid_ys = actions_df['grid_y'].values
    directions = actions_df['direction'].values

    for i in range(len(actions)):
        exp_pos = (int(grid_xs[i]), int(grid_ys[i]))
        exp_dir = int(directions[i])
        act_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]))
        act_dir = int(env.agent_dir)

        if exp_pos != act_pos or exp_dir != act_dir:
            return ('DIVERGE', f'step {i}/{len(actions)} '
                    f'expected pos={exp_pos} dir={exp_dir}, '
                    f'env pos={act_pos} dir={act_dir} tc={env.trial_count}')

        obs, reward, terminated, truncated, info = env.step(int(actions[i]))
        if terminated or truncated:
            return ('OK', f'ended {i}/{len(actions)}')

    return ('OK', f'all {len(actions)} steps')


def check_session(parquet_path):
    """Run headless divergence check on a per-session parquet (legacy mode).

    Uses the legacy hardcoded `'PI+VC f2 acquisition'` paradigm — only valid
    for the originally-tested PI+VC subjects. For other groups (PI, VC,
    PI+VC_f1), use ``--from-dataset`` mode instead.
    """
    actions_df, meta = load_yoked_parquet(parquet_path)
    session_phase = meta.get('session_phase', '')
    session_number = meta.get('session_number', '')
    orientation = meta.get('cue_goal_orientation', 'N/NE')

    is_exposure = session_phase == 'Exposure'
    trial_configs = None
    if 'trial_configs' in meta:
        trial_configs = json.loads(meta['trial_configs'])

    goal_location = 'NE'
    if trial_configs:
        goal_location = _GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE')

    if is_exposure:
        n_rewards = int(meta.get('n_rewards', '0'))
        env_type = 'exposure_b' if (session_number == '2e' and n_rewards > 0) else 'exposure'
    else:
        env_type = 'PI+VC f2 acquisition'

    env = CornerMazeEnv(
        render_mode='rgb_array',
        max_steps=max(len(actions_df) * 2, 10000),
        session_type=env_type,
        agent_cue_goal_orientation=orientation,
        start_goal_location=goal_location,
    )
    env.reset()

    if trial_configs:
        _inject_trial_configs(env, trial_configs)
    else:
        init_pos = (int(actions_df['grid_x'].iloc[0]), int(actions_df['grid_y'].iloc[0]))
        init_dir = int(actions_df['direction'].iloc[0])
        env.agent_pos = init_pos
        env.agent_dir = init_dir
        env.agent_pose = (*init_pos, init_dir)
        env.fwd_pos = env.front_pos
        env.fwd_cell = env.grid.get(*env.fwd_pos)

    try:
        return _run_replay(env, actions_df)
    finally:
        env.close()


def check_session_from_dataset(sess_row, subj_row, actions_df):
    """Run headless divergence check using the normalized dataset.

    Looks up the env paradigm by (training_group, session_type) via
    ``PARADIGM_MAP``; passes ``trial_configs`` so the env's trial sequence
    matches the rat's recorded sequence step-for-step. Falls back to the
    exposure paradigm for ``session_phase == 'Exposure'``.
    """
    session_phase = sess_row['session_phase']
    is_exposure = session_phase == 'Exposure'

    trial_configs = None
    raw = sess_row.get('trial_configs')
    if isinstance(raw, str) and raw:
        try:
            trial_configs = json.loads(raw)
            if not trial_configs:
                trial_configs = None
        except json.JSONDecodeError:
            trial_configs = None

    if is_exposure:
        n_rewards = int(sess_row.get('n_rewards', 0))
        env_type = 'exposure_b' if (sess_row['session_number'] == '2e' and n_rewards > 0) else 'exposure'
        goal_location = 'NE'
        if trial_configs:
            goal_location = _GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE')
        env = CornerMazeEnv(
            render_mode='rgb_array',
            max_steps=max(len(actions_df) * 2, 10000),
            session_type=env_type,
            agent_cue_goal_orientation=subj_row['cue_goal_orientation'],
            start_goal_location=goal_location,
        )
        env.reset()
        if trial_configs:
            _inject_trial_configs(env, trial_configs)
        else:
            init_pos = (int(actions_df['grid_x'].iloc[0]), int(actions_df['grid_y'].iloc[0]))
            init_dir = int(actions_df['direction'].iloc[0])
            env.agent_pos = init_pos
            env.agent_dir = init_dir
            env.agent_pose = (*init_pos, init_dir)
            env.fwd_pos = env.front_pos
            env.fwd_cell = env.grid.get(*env.fwd_pos)
    else:
        kw = map_session_to_env_kwargs(
            training_group=subj_row['training_group'],
            yoked_session_type=sess_row['session_type'],
            cue_goal_orientation=subj_row['cue_goal_orientation'],
            start_goal_location=(_GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE')
                                 if trial_configs else None),
            trial_configs=trial_configs,
        )
        if kw is None:
            return ('SKIP', f'unmapped (group={subj_row["training_group"]!r}, '
                    f'type={sess_row["session_type"]!r})')
        env = CornerMazeEnv(
            render_mode='rgb_array',
            max_steps=max(len(actions_df) * 2, 10000),
            **kw,
        )
        env.reset()

    try:
        return _run_replay(env, actions_df)
    finally:
        env.close()


def _load_dataset(dataset_dir):
    """Load the 5-table dataset and the three per-phase action tables."""
    sessions = pd.read_parquet(os.path.join(dataset_dir, 'sessions.parquet'))
    subjects = pd.read_parquet(os.path.join(dataset_dir, 'subjects.parquet'))
    actions = {
        'synthetic_pretrial': pd.read_parquet(
            os.path.join(dataset_dir, 'actions_synthetic_pretrial.parquet')),
        'real_pretrial':      pd.read_parquet(
            os.path.join(dataset_dir, 'actions_real_pretrial.parquet'))
            if os.path.exists(os.path.join(dataset_dir, 'actions_real_pretrial.parquet')) else None,
        'exposure':           pd.read_parquet(
            os.path.join(dataset_dir, 'actions_exposure.parquet'))
            if os.path.exists(os.path.join(dataset_dir, 'actions_exposure.parquet')) else None,
    }
    return sessions, subjects, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=None,
                        help='Filter to a single subject by name.')
    parser.add_argument('--dir', type=str, default='data/yoked',
                        help='(Legacy mode) directory containing per-session parquets.')
    parser.add_argument('--from-dataset', action='store_true',
                        help='Read from the normalized 5-table dataset under '
                             '--dataset-dir instead of per-session parquets.')
    parser.add_argument('--dataset-dir', type=str, default='data/yoked/dataset',
                        help='Path to the normalized dataset (used with --from-dataset).')
    parser.add_argument('--actions-variant', type=str, default='synthetic_pretrial',
                        choices=['synthetic_pretrial', 'real_pretrial'],
                        help='Which Acquisition action stream to check (Exposure '
                             'always uses actions_exposure.parquet).')
    parser.add_argument('--phase', type=str, default=None,
                        choices=['Acquisition', 'Exposure'],
                        help='Filter to a single phase.')
    args = parser.parse_args()

    if args.from_dataset:
        sessions, subjects, actions = _load_dataset(args.dataset_dir)
        if args.subject:
            subj_match = subjects[subjects['subject_name'] == args.subject]
            if len(subj_match) == 0:
                print(f'No subject named {args.subject!r} in dataset.')
                return
            sessions = sessions[sessions['subject_id'].isin(subj_match['subject_id'])]
        if args.phase:
            sessions = sessions[sessions['session_phase'] == args.phase]
        sessions = sessions.sort_values(['subject_id', 'session_number']).reset_index(drop=True)

        if len(sessions) == 0:
            print('No sessions matched filters.')
            return

        n_ok = 0
        n_diverge = 0
        n_skip = 0
        failures = []

        for _, sess in sessions.iterrows():
            subj = subjects[subjects['subject_id'] == sess['subject_id']].iloc[0]
            if sess['session_phase'] == 'Exposure':
                tbl = actions['exposure']
            else:
                tbl = actions[args.actions_variant]
            if tbl is None:
                n_skip += 1
                continue
            actions_df = tbl[tbl['session_id'] == sess['session_id']].sort_values('step')
            if len(actions_df) == 0:
                n_skip += 1
                continue
            label = f'{subj["subject_name"]}_{sess["session_number"]}_{sess["session_phase"][:3]}'
            try:
                status, detail = check_session_from_dataset(sess, subj, actions_df)
            except Exception as e:
                status, detail = ('ERR', str(e))
            if status == 'OK':
                n_ok += 1
                print(f'  OK   {label}: {detail}')
            elif status == 'SKIP':
                n_skip += 1
                print(f'  SKIP {label}: {detail}')
            else:
                n_diverge += 1
                failures.append((label, detail))
                print(f'  FAIL {label}: {detail}')

        print(f'\n{n_ok} ok, {n_diverge} diverge, {n_skip} skipped '
              f'({len(sessions)} sessions total)')
        if failures:
            print('\nFailed sessions:')
            for label, detail in failures:
                print(f'  {label}: {detail}')
        return

    # Legacy per-session-parquet mode
    pattern = os.path.join(args.dir, '*.parquet')
    files = sorted(glob(pattern))
    if args.subject:
        files = [f for f in files if os.path.basename(f).startswith(args.subject + '_')]
    if not files:
        print('No per-session parquet files found. Did you mean --from-dataset?')
        return

    n_ok = 0
    n_fail = 0
    failures = []
    for fpath in files:
        name = os.path.basename(fpath).replace('.parquet', '')
        try:
            status, detail = check_session(fpath)
            if status == 'OK':
                n_ok += 1
                print(f'  OK   {name}: {detail}')
            else:
                n_fail += 1
                failures.append((name, detail))
                print(f'  FAIL {name}: {detail}')
        except Exception as e:
            n_fail += 1
            failures.append((name, str(e)))
            print(f'  ERR  {name}: {e}')

    print(f'\n{n_ok} ok, {n_fail} failed out of {len(files)} sessions')
    if failures:
        print('\nFailed sessions:')
        for name, detail in failures:
            print(f'  {name}: {detail}')


if __name__ == '__main__':
    main()
