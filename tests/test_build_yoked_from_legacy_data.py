"""Tests for ``scripts/build_yoked_from_legacy_data.py``.

The converter lives in the top-level ``scripts/`` dir (one-shot tool,
not part of the importable package), so we load it via importlib.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Load the script as an ad-hoc module
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_yoked_from_legacy_data.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_yoked_from_legacy_data", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def conv():
    return _load()


# ---------------------------------------------------------------------------
# Synthetic source-data helpers
# ---------------------------------------------------------------------------

LEGACY_COLUMNS = ["filestem", "X", "Y", "D", "A", "B", "R"]


def _make_legacy_session_df(n: int = 10, actions=None) -> pd.DataFrame:
    """Build a fake per-session legacy parquet with the expected schema."""
    actions = actions if actions is not None else [0, 1, 2, 5, 6] * ((n + 4) // 5)
    actions = list(actions)[:n]
    return pd.DataFrame({
        "filestem": ["expa_x_x_xx"] * n,
        "X":        np.arange(n, dtype=np.float64),
        "Y":        np.arange(n, dtype=np.float64),
        "D":        np.zeros(n, dtype=np.float64),
        "A":        np.array(actions, dtype=np.float64),
        "B":        np.arange(1, n + 1, dtype=np.float64),
        "R":        np.arange(n, 0, -1, dtype=np.float64),
    })


def _write_legacy_session(root: Path, rat: str, suffix: str, df: pd.DataFrame) -> Path:
    rat_dir = root / "data" / "parquet" / "sessions" / rat
    rat_dir.mkdir(parents=True, exist_ok=True)
    p = rat_dir / f"{rat}_{suffix}.parquet"
    df.to_parquet(p, engine="pyarrow")
    return p


# ---------------------------------------------------------------------------
# Action remap
# ---------------------------------------------------------------------------

def test_action_map_known_values(conv):
    assert conv.ACTION_MAP == {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}


def test_action_remap_via_pandas_default_to_pause(conv):
    """Values outside the ACTION_MAP keys must default to 4 (pause)."""
    raw = pd.Series([0, 1, 2, 5, 6, 99, np.nan])
    mapped = raw.map(conv.ACTION_MAP).fillna(4).astype(int)
    assert mapped.tolist() == [0, 1, 2, 3, 4, 4, 4]


# ---------------------------------------------------------------------------
# Filename → session_type
# ---------------------------------------------------------------------------

def test_session_type_for_pre1(conv):
    assert conv.session_type_for("CM001_pre1.parquet") == "exposure"


def test_session_type_for_pre2(conv):
    assert conv.session_type_for("CM042_pre2.parquet") == "exposure_b"


# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------

def test_deterministic_seed_is_stable(conv):
    s1 = conv.deterministic_seed("CM001", "exposure")
    s2 = conv.deterministic_seed("CM001", "exposure")
    assert s1 == s2


def test_deterministic_seed_varies_by_input(conv):
    a = conv.deterministic_seed("CM001", "exposure")
    b = conv.deterministic_seed("CM001", "exposure_b")
    c = conv.deterministic_seed("CM002", "exposure")
    assert a != b
    assert a != c


def test_deterministic_seed_fits_int32(conv):
    """6 hex chars → fits in 24 bits; comfortably within int range."""
    s = conv.deterministic_seed("CM001", "exposure")
    assert isinstance(s, int)
    assert 0 <= s < (1 << 24)


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------

def test_discover_finds_pre1_and_pre2(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    _write_legacy_session(tmp_path, "CM001", "pre2", _make_legacy_session_df(5))
    _write_legacy_session(tmp_path, "CM002", "pre2", _make_legacy_session_df(5))

    records = conv.discover(tmp_path)
    rats_types = sorted((r, st) for r, st, _ in records)
    assert rats_types == [
        ("CM001", "exposure"),
        ("CM001", "exposure_b"),
        ("CM002", "exposure_b"),
    ]


def test_discover_ignores_unrelated_files(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    # Junk file the glob shouldn't match: pre3, summary, .csv, etc.
    rat_dir = tmp_path / "data" / "parquet" / "sessions" / "CM001"
    _make_legacy_session_df(5).to_parquet(rat_dir / "CM001_pre3.parquet")
    _make_legacy_session_df(5).to_parquet(rat_dir / "CM001_summary.parquet")
    (rat_dir / "notes.txt").write_text("hello")

    records = conv.discover(tmp_path)
    assert len(records) == 1
    assert records[0][1] == "exposure"


def test_discover_raises_on_missing_root(conv, tmp_path):
    with pytest.raises(FileNotFoundError):
        conv.discover(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# build_subjects()
# ---------------------------------------------------------------------------

def test_build_subjects_schema_and_ids(conv):
    rats = ["CM001", "CM003", "CM005"]
    df = conv.build_subjects(rats)
    assert list(df.columns) == [
        "subject_id", "subject_name", "training_group", "cue_goal_orientation"
    ]
    assert df["subject_id"].tolist() == [0, 1, 2]
    assert df["subject_name"].tolist() == rats


def test_build_subjects_placeholders(conv):
    df = conv.build_subjects(["CM001"])
    assert df["training_group"].iloc[0] == conv.TRAINING_GROUP_PLACEHOLDER
    assert df["cue_goal_orientation"].iloc[0] == conv.ORIENTATION_PLACEHOLDER


# ---------------------------------------------------------------------------
# build_sessions_and_actions()
# ---------------------------------------------------------------------------

def test_build_sessions_pre1_pre2_session_types(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    _write_legacy_session(tmp_path, "CM001", "pre2", _make_legacy_session_df(5))
    records = conv.discover(tmp_path)
    sessions, _ = conv.build_sessions_and_actions(records, {"CM001": 0})

    by_num = sessions.set_index("session_number")["session_type"].to_dict()
    assert by_num == {1: "exposure", 2: "exposure_b"}


def test_build_actions_remaps_action_codes(conv, tmp_path):
    df_in = _make_legacy_session_df(n=6, actions=[0, 1, 2, 5, 6, 99])
    _write_legacy_session(tmp_path, "CM001", "pre1", df_in)
    records = conv.discover(tmp_path)
    _, actions = conv.build_sessions_and_actions(records, {"CM001": 0})

    assert actions["action"].tolist() == [0, 1, 2, 3, 4, 4]
    assert actions["action"].dtype == np.int64


def test_build_actions_drops_legacy_columns(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    records = conv.discover(tmp_path)
    _, actions = conv.build_sessions_and_actions(records, {"CM001": 0})
    assert "B" not in actions.columns
    assert "R" not in actions.columns
    assert "filestem" not in actions.columns
    assert set(actions.columns) >= {
        "session_id", "step", "action", "grid_x", "grid_y", "direction"
    }


def test_build_skips_empty_session_files(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    _write_legacy_session(
        tmp_path, "CM002", "pre1",
        pd.DataFrame({c: pd.array([], dtype="float64") for c in LEGACY_COLUMNS}),
    )
    records = conv.discover(tmp_path)
    sessions, actions = conv.build_sessions_and_actions(
        records, {"CM001": 0, "CM002": 1}
    )
    assert len(sessions) == 1
    assert sessions["subject_id"].tolist() == [0]
    # action rows belong to the surviving session only
    assert (actions["session_id"] == 0).all()


def test_build_sessions_assigns_running_session_ids(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(3))
    _write_legacy_session(tmp_path, "CM001", "pre2", _make_legacy_session_df(3))
    _write_legacy_session(tmp_path, "CM002", "pre1", _make_legacy_session_df(3))
    records = conv.discover(tmp_path)
    sessions, actions = conv.build_sessions_and_actions(
        records, {"CM001": 0, "CM002": 1}
    )
    assert sessions["session_id"].tolist() == [0, 1, 2]
    assert sorted(actions["session_id"].unique().tolist()) == [0, 1, 2]


def test_build_sessions_step_starts_at_zero(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(4))
    records = conv.discover(tmp_path)
    _, actions = conv.build_sessions_and_actions(records, {"CM001": 0})
    assert actions["step"].tolist() == [0, 1, 2, 3]


def test_build_sessions_trial_configs_is_empty_json_list(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(3))
    records = conv.discover(tmp_path)
    sessions, _ = conv.build_sessions_and_actions(records, {"CM001": 0})
    # Must round-trip through json.loads as an empty list — that's what
    # build_returns_dataset._first_trial_goal expects.
    import json
    parsed = json.loads(sessions["trial_configs"].iloc[0])
    assert parsed == []


# ---------------------------------------------------------------------------
# End-to-end build()
# ---------------------------------------------------------------------------

def test_build_writes_three_parquets_with_expected_schema(conv, tmp_path):
    _write_legacy_session(tmp_path, "CM001", "pre1", _make_legacy_session_df(5))
    _write_legacy_session(tmp_path, "CM001", "pre2", _make_legacy_session_df(5))
    _write_legacy_session(tmp_path, "CM002", "pre2", _make_legacy_session_df(5))

    out = tmp_path / "out"
    conv.build(tmp_path, out)

    subjects = pd.read_parquet(out / "subjects.parquet")
    sessions = pd.read_parquet(out / "sessions.parquet")
    actions = pd.read_parquet(out / "actions_real_pretrial.parquet")

    assert len(subjects) == 2
    assert len(sessions) == 3
    assert len(actions) == 15
    assert set(actions["session_id"].unique().tolist()) == {0, 1, 2}
    assert set(sessions["session_type"].unique()) == {"exposure", "exposure_b"}


def test_build_raises_when_no_sessions_discovered(conv, tmp_path):
    """Empty-but-existent sessions root: discover returns [] → build raises."""
    (tmp_path / "data" / "parquet" / "sessions").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no session parquets"):
        conv.build(tmp_path, tmp_path / "out")
