"""One-time pre-processing: replay yoked sessions through CornerMazeEnv to
attach per-step ``reward`` and per-trial-with-ITI-start ``return_to_go``.

Reads:
  data/yoked/dataset/subjects.parquet
  data/yoked/dataset/sessions.parquet
  data/yoked/dataset/actions_synthetic_pretrial.parquet  (default)

Writes:
  data/yoked/dataset/actions_with_returns.parquet

Sessions whose (training_group, yoked_session_type) is unmapped (plan §13
TODO cell: VC × Dark Train) are skipped with a warning.

KNOWN LIMITATION (post-Phase 1.5 follow-up):
  The yoked action stream covers the FULL behavioural session (exposure
  A + exposure B + acquisition trials), but env paradigms like
  "PI+VC f2 novel route" model only the trial-cycle portion. As a result,
  the early portion of every yoked session is replayed through a
  pretrial/trial config rather than the matching exposure config; reward
  attribution is sparser than the rat's dataset-recorded reward count.
  Smoke runs match dataset rewards on roughly half of sessions and
  partially on the rest. RTG arithmetic is correct on whatever rewards
  the env DOES emit — the gap is a paradigm-mapping issue, not an RTG bug.
  See md/dt-repo-plan.md §11 / §16.2 for context on the legacy yoking
  pipeline coverage.

Usage:
  python -m corner_maze_rl.scripts.build_returns_dataset \
      --dataset-dir data/yoked/dataset \
      --out data/yoked/dataset/actions_with_returns.parquet \
      [--limit-subjects 3]   # for smoke runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Goal-corner index to compass direction (inverse of GOAL_LOCATION_MAP).
GOAL_IDX_TO_DIR = {0: "NE", 1: "SE", 2: "SW", 3: "NW"}


def _first_trial_goal(trial_configs_blob) -> str | None:
    """Parse the first trial's goal corner from a session's trial_configs.

    trial_configs is stored as a JSON string of a list of [arm, cue, goal, tag]
    tuples. Returns the compass-direction string (e.g. 'NE') for the first
    trial's goal, or None if the blob can't be parsed.
    """
    try:
        if isinstance(trial_configs_blob, str):
            data = json.loads(trial_configs_blob)
        else:
            data = trial_configs_blob
        if not data:
            return None
        first = data[0]
        goal_idx = int(first[2])
        return GOAL_IDX_TO_DIR.get(goal_idx)
    except (ValueError, TypeError, IndexError, json.JSONDecodeError):
        return None

from corner_maze_rl.data.compute_returns import compute_returns_for_session
from corner_maze_rl.data.load import (
    YokedPaths,
    load_sessions,
    load_subjects,
    load_actions_for_session,
)
from corner_maze_rl.data.session_types import map_session_to_env_kwargs
from corner_maze_rl.env.corner_maze_env import CornerMazeEnv


def _build_env_factory(
    *,
    training_group: str,
    yoked_session_type: str,
    cue_goal_orientation: str,
    start_goal_location: str | None = None,
    trial_configs: list | None = None,
):
    """Return a zero-arg env factory or None if unmapped."""
    kwargs = map_session_to_env_kwargs(
        training_group=training_group,
        yoked_session_type=yoked_session_type,
        cue_goal_orientation=cue_goal_orientation,
        start_goal_location=start_goal_location,
        trial_configs=trial_configs,
    )
    if kwargs is None:
        return None

    def factory():
        return CornerMazeEnv(**kwargs)

    return factory


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="data/yoked/dataset",
                   help="dir containing subjects/sessions/actions parquets")
    p.add_argument("--out", default="data/yoked/dataset/actions_with_returns.parquet",
                   help="output parquet path")
    p.add_argument("--actions-variant", default="synthetic_pretrial",
                   choices=["synthetic_pretrial", "real_pretrial", "exposure"],
                   help="which actions_*.parquet to consume. 'exposure' is "
                        "available but currently unused — session_types.py has "
                        "no Exposure paradigm mapping yet (plan §11 follow-up).")
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="process only the first N subjects (for smoke testing)")
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="process only the first N sessions per subject")
    p.add_argument("--seed-default", type=int, default=42,
                   help="env reset seed when sessions table doesn't supply one")
    args = p.parse_args(argv)

    paths = YokedPaths.from_dir(args.dataset_dir, actions_variant=args.actions_variant)
    paths.assert_exist()

    subjects = load_subjects(paths)
    sessions = load_sessions(paths)

    if args.limit_subjects is not None:
        keep = subjects["subject_name"].iloc[: args.limit_subjects]
        subjects = subjects[subjects["subject_name"].isin(keep)]
        sessions = sessions[sessions["subject_id"].isin(subjects["subject_id"])]

    skipped_unmapped: list[tuple[str, str]] = []
    failed: list[tuple[int, str]] = []
    chunks: list[pd.DataFrame] = []

    t0 = time.time()
    grand_total = 0

    subj_iter = subjects.iterrows()
    pbar = tqdm(total=len(sessions), desc="sessions")
    for _, subj in subj_iter:
        subj_name = str(subj["subject_name"])

        subj_sessions = sessions[sessions["subject_id"] == subj["subject_id"]] \
            .sort_values("session_number")
        if args.limit_sessions is not None:
            subj_sessions = subj_sessions.head(args.limit_sessions)

        for _, sess in subj_sessions.iterrows():
            pbar.update(1)
            raw_tc = sess["trial_configs"]
            try:
                parsed_tc = json.loads(raw_tc) if isinstance(raw_tc, str) else list(raw_tc)
            except (TypeError, json.JSONDecodeError):
                parsed_tc = None
            factory = _build_env_factory(
                training_group=str(subj["training_group"]),
                yoked_session_type=str(sess["session_type"]),
                cue_goal_orientation=str(subj["cue_goal_orientation"]),
                start_goal_location=_first_trial_goal(sess["trial_configs"]),
                trial_configs=parsed_tc,
            )
            if factory is None:
                skipped_unmapped.append(
                    (str(subj["training_group"]), str(sess["session_type"]))
                )
                continue

            try:
                actions = load_actions_for_session(paths, int(sess["session_id"]))
                seed = int(sess.get("seed", args.seed_default))
                out = compute_returns_for_session(actions, factory, seed=seed)
                # Carry session_id/subject metadata for joins.
                out["subject_name"] = subj_name
                out["training_group"] = str(subj["training_group"])
                out["yoked_session_type"] = str(sess["session_type"])
                chunks.append(out)
                grand_total += len(out)
            except Exception as e:  # noqa: BLE001
                failed.append((int(sess["session_id"]), repr(e)))

    pbar.close()
    dt = time.time() - t0

    if not chunks:
        print("No sessions produced output.", file=sys.stderr)
        return 1

    big = pd.concat(chunks, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out_path, engine="pyarrow")

    print()
    print(f"Wrote {grand_total} rows to {out_path}")
    print(f"  elapsed: {dt:.1f}s")
    print(f"  unmapped (group, session_type) skips: "
          f"{len(skipped_unmapped)} (unique pairs: "
          f"{len(set(skipped_unmapped))})")
    if skipped_unmapped:
        for pair in sorted(set(skipped_unmapped)):
            n = skipped_unmapped.count(pair)
            print(f"    {pair}: {n} sessions skipped")
    if failed:
        print(f"  FAILED: {len(failed)} sessions")
        for sid, err in failed[:10]:
            print(f"    session_id={sid}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
