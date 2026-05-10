"""Convert ../rat-neural-model-2 per-session parquets to the 3-table yoked form.

Source layout
-------------
    <LEGACY_DATA>/data/parquet/sessions/CM<id>/CM<id>_pre1.parquet  (exposure A)
    <LEGACY_DATA>/data/parquet/sessions/CM<id>/CM<id>_pre2.parquet  (exposure B)

Per-row columns: [filestem, X, Y, D, A, B, R].

Output layout (matches data/load.py expectations)
-------------------------------------------------
    data/yoked/dataset/subjects.parquet
    data/yoked/dataset/sessions.parquet
    data/yoked/dataset/actions_real_pretrial.parquet

Notes
-----
- Source has only exposure A/B sessions — no acquisition or probe data. The
  resulting yoked dataset is exposure-only; downstream env paradigms reduce
  to "exposure" / "exposure_b".
- training_group="PI" / cue_goal_orientation="NS" are placeholders. The
  exposure paradigm doesn't read them; if non-exposure data is added later
  the placeholders will need real values.
- Action remap: {0:0, 1:1, 2:2, 5:3, 6:4} (default → 4 = pause). Matches
  rat-neural-model-2/notebooks/DTtrainer.ipynb.
- The legacy B/R columns are dropped; compute_returns.py recomputes the
  real reward signal by replaying actions through the env.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_MAP = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
TRAINING_GROUP_PLACEHOLDER = "PI"
ORIENTATION_PLACEHOLDER = "NS"


def session_type_for(filename: str) -> str:
    return "exposure" if filename.endswith("_pre1.parquet") else "exposure_b"


def deterministic_seed(name: str, stype: str) -> int:
    h = hashlib.md5(f"{name}|{stype}".encode()).hexdigest()
    return int(h[:6], 16)


def discover(legacy_data_root: Path) -> list[tuple[str, str, Path]]:
    sessions_root = legacy_data_root / "data" / "parquet" / "sessions"
    if not sessions_root.is_dir():
        raise FileNotFoundError(f"sessions dir not found: {sessions_root}")
    out: list[tuple[str, str, Path]] = []
    for rat_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        for f in sorted(rat_dir.glob("CM*_pre[12].parquet")):
            out.append((rat_dir.name, session_type_for(f.name), f))
    return out


def build_subjects(rats: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id":           list(range(len(rats))),
        "subject_name":         rats,
        "training_group":       [TRAINING_GROUP_PLACEHOLDER] * len(rats),
        "cue_goal_orientation": [ORIENTATION_PLACEHOLDER] * len(rats),
    })


def build_sessions_and_actions(
    records: list[tuple[str, str, Path]],
    subj_idx: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sess_rows: list[dict] = []
    action_chunks: list[pd.DataFrame] = []
    sid = 0
    for rat, stype, path in records:
        df = pd.read_parquet(path)
        if len(df) == 0:
            print(f"  skip empty: {rat}/{path.name}")
            continue
        sess_rows.append({
            "session_id":     sid,
            "subject_id":     subj_idx[rat],
            "session_number": 1 if stype == "exposure" else 2,
            "session_type":   stype,
            "trial_configs":  "[]",
            "seed":           deterministic_seed(rat, stype),
        })
        action_chunks.append(pd.DataFrame({
            "session_id": sid,
            "step":       np.arange(len(df), dtype=np.int64),
            "action":     df["A"].map(ACTION_MAP).fillna(4).astype(np.int64),
            "grid_x":     df["X"].astype(np.int32),
            "grid_y":     df["Y"].astype(np.int32),
            "direction":  df["D"].astype(np.int32),
        }))
        sid += 1
    sessions = pd.DataFrame(sess_rows)
    actions = pd.concat(action_chunks, ignore_index=True) if action_chunks else pd.DataFrame()
    return sessions, actions


def build(legacy_data_root: Path, out_dir: Path) -> None:
    records = discover(legacy_data_root)
    if not records:
        raise RuntimeError(f"no session parquets discovered under {legacy_data_root}")

    rats = sorted({r for r, _, _ in records})
    subjects = build_subjects(rats)
    subj_idx = dict(zip(subjects["subject_name"], subjects["subject_id"]))

    sessions, actions = build_sessions_and_actions(records, subj_idx)

    out_dir.mkdir(parents=True, exist_ok=True)
    subjects.to_parquet(out_dir / "subjects.parquet", engine="pyarrow")
    sessions.to_parquet(out_dir / "sessions.parquet", engine="pyarrow")
    actions.to_parquet(out_dir / "actions_real_pretrial.parquet", engine="pyarrow")

    print()
    print(f"  subjects: {len(subjects)}  -> {out_dir / 'subjects.parquet'}")
    print(f"  sessions: {len(sessions)}  -> {out_dir / 'sessions.parquet'}")
    print(f"  action rows: {len(actions):,}  "
          f"-> {out_dir / 'actions_real_pretrial.parquet'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-data-root", default="../rat-neural-model-2",
                   help="path to rat-neural-model-2 checkout")
    p.add_argument("--out", default="data/yoked/dataset",
                   help="output dir for the 3-table yoked dataset")
    args = p.parse_args()
    build(Path(args.legacy_data_root), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
