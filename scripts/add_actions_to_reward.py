"""Add ``actions_to_reward`` column to existing ``data/yoked/dataset/actions_*.parquet``.

One-shot post-process for datasets aggregated before ``build_dataset.py`` was
taught to compute the column. Idempotent: running twice produces the same
output. After this lands, ``corner-maze-build-dataset`` does the same work
automatically on every rebuild, so this script becomes a no-op going forward.

The column is the count of action steps remaining until the next correct
PICKUP (``action == 3 & rewarded == 1``), zero at the PICKUP itself. Each
session's count is computed independently — windows do not span session
boundaries.

Usage:
    python scripts/add_actions_to_reward.py
    python scripts/add_actions_to_reward.py --dataset-dir data/yoked/dataset
"""
from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np
import pandas as pd

from corner_maze_rl.yoking.build_dataset import compute_actions_to_reward


ACTION_TABLES = (
    "actions_synthetic_pretrial.parquet",
    "actions_real_pretrial.parquet",
    "actions_exposure.parquet",
)


def add_column(path: str) -> tuple[int, int]:
    """Read ``path``, compute ``actions_to_reward`` per session, write back.

    Returns ``(n_sessions, n_rows)``. Idempotent: if the column already
    exists with correct values, the rewritten parquet is identical.
    """
    df = duckdb.sql(f"SELECT * FROM '{path}' ORDER BY session_id, step").fetchdf()
    out_parts: list[pd.DataFrame] = []
    for session_id, group in df.groupby("session_id", sort=False):
        group = group.copy()
        group["actions_to_reward"] = compute_actions_to_reward(
            group["action"].to_numpy(),
            group["rewarded"].to_numpy(),
        )
        out_parts.append(group)
    out = pd.concat(out_parts, ignore_index=True)
    # Preserve the column order used by build_dataset.py.
    cols = ["session_id", "step", "action", "grid_x", "grid_y",
            "direction", "rewarded", "actions_to_reward"]
    out = out[cols]
    out.to_parquet(path, index=False)
    return out["session_id"].nunique(), len(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add actions_to_reward column to existing aggregated parquets.",
    )
    parser.add_argument("--dataset-dir", default="data/yoked/dataset",
                        help="Directory containing aggregated action parquets.")
    args = parser.parse_args()

    for fname in ACTION_TABLES:
        path = os.path.join(args.dataset_dir, fname)
        if not os.path.exists(path):
            print(f"  skip {fname}: not present")
            continue
        n_sess, n_rows = add_column(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  ok   {fname}: {n_sess} sessions, {n_rows} rows, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
