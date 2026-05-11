"""Invariants on data/yoked/dataset/ that DT training relies on.

Skipped when the dataset isn't materialized (e.g. fresh clone). Run after
``corner-maze-build-dataset`` to lock in the contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "yoked" / "dataset"

ACTION_TABLES = (
    "actions_synthetic_pretrial",
    "actions_real_pretrial",
    "actions_exposure",
)


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} not present; run corner-maze-build-dataset")
    return path


@pytest.mark.parametrize("table", ACTION_TABLES)
def test_every_session_ends_at_rewarded_pickup(table):
    """Last row of each session in every action table must be a rewarded PICKUP.

    DT per-trial RTG depends on this: no dangling pretrial, no aborted
    trial-nav, no wrong-well-only final trial.
    """
    path = _require(DATASET_DIR / f"{table}.parquet")
    last_rows = duckdb.sql(
        f"""
        SELECT session_id, action, rewarded
        FROM '{path}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY step DESC) = 1
        """
    ).fetchdf()
    bad_reward = last_rows[last_rows["rewarded"] != 1]
    bad_action = last_rows[last_rows["action"] != 3]  # ACT_PICKUP
    assert bad_reward.empty, (
        f"{table}: {len(bad_reward)} sessions whose last row is not rewarded: "
        f"{bad_reward['session_id'].tolist()}"
    )
    assert bad_action.empty, (
        f"{table}: {len(bad_action)} sessions whose last action is not PICKUP(3): "
        f"{bad_action['session_id'].tolist()}"
    )


def test_acquisition_trial_counts_consistent():
    """For every Acquisition row in sessions.parquet:
    n_trials == n_rewards == len(trial_configs).
    """
    sessions_path = _require(DATASET_DIR / "sessions.parquet")
    df = duckdb.sql(
        f"""
        SELECT session_id, session_phase, n_trials, n_rewards, trial_configs
        FROM '{sessions_path}'
        WHERE session_phase = 'Acquisition'
        """
    ).fetchdf()
    bad = []
    for _, row in df.iterrows():
        configs = json.loads(row["trial_configs"]) if row["trial_configs"] else []
        if not (row["n_trials"] == row["n_rewards"] == len(configs)):
            bad.append(
                (
                    int(row["session_id"]),
                    int(row["n_trials"]),
                    int(row["n_rewards"]),
                    len(configs),
                )
            )
    assert not bad, (
        f"{len(bad)} Acquisition sessions with n_trials/n_rewards/len(trial_configs) "
        f"mismatch (session_id, n_trials, n_rewards, len_configs): {bad[:10]}"
    )


def test_no_zero_reward_sessions_in_action_tables():
    """Truncation drops zero-reward sessions; ensure none slipped into the tables."""
    for table in ACTION_TABLES:
        path = _require(DATASET_DIR / f"{table}.parquet")
        zeros = duckdb.sql(
            f"""
            SELECT session_id, SUM(rewarded) AS r
            FROM '{path}'
            GROUP BY session_id
            HAVING SUM(rewarded) = 0
            """
        ).fetchdf()
        assert zeros.empty, (
            f"{table}: zero-reward sessions present: {zeros['session_id'].tolist()}"
        )


# ---------------------------------------------------------------------------
# actions_to_reward column invariants (cost-agnostic structural countdown)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", ACTION_TABLES)
def test_actions_to_reward_dtype_integer(table):
    """The actions_to_reward column must be integer-typed."""
    path = _require(DATASET_DIR / f"{table}.parquet")
    dtype_row = duckdb.sql(
        f"""
        SELECT column_type FROM (DESCRIBE SELECT * FROM '{path}')
        WHERE column_name = 'actions_to_reward'
        """
    ).fetchone()
    assert dtype_row is not None, f"{table}: missing actions_to_reward column"
    assert "INT" in dtype_row[0].upper(), (
        f"{table}: actions_to_reward must be integer, got {dtype_row[0]!r}"
    )


@pytest.mark.parametrize("table", ACTION_TABLES)
def test_actions_to_reward_zero_at_session_end(table):
    """Last row of every session must have actions_to_reward == 0."""
    path = _require(DATASET_DIR / f"{table}.parquet")
    bad = duckdb.sql(
        f"""
        SELECT session_id, actions_to_reward
        FROM '{path}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY step DESC) = 1
        """
    ).fetchdf()
    bad = bad[bad["actions_to_reward"] != 0]
    assert bad.empty, (
        f"{table}: {len(bad)} sessions whose last row has actions_to_reward != 0: "
        f"{bad['session_id'].tolist()[:5]}"
    )


@pytest.mark.parametrize("table", ACTION_TABLES)
def test_actions_to_reward_zero_iff_correct_pickup(table):
    """actions_to_reward == 0 ⇔ (action == 3 AND rewarded == 1)."""
    path = _require(DATASET_DIR / f"{table}.parquet")
    mismatch = duckdb.sql(
        f"""
        SELECT session_id, step, action, rewarded, actions_to_reward
        FROM '{path}'
        WHERE (actions_to_reward = 0) != (action = 3 AND rewarded = 1)
        LIMIT 5
        """
    ).fetchdf()
    assert mismatch.empty, (
        f"{table}: {len(mismatch)} rows violate "
        f"actions_to_reward == 0 ⇔ correct PICKUP. Examples:\n{mismatch.to_string(index=False)}"
    )
