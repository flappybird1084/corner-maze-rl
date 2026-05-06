"""Tests for corner_maze_rl.utils.run_io.

Covers seeding determinism, run config writing (with auto-fields), git sha
capture, and dataset hashing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from corner_maze_rl.utils.run_io import (
    capture_git_dirty,
    capture_git_sha,
    generate_seed,
    hash_dataset,
    load_run_config,
    save_run_config,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def test_set_global_seed_makes_random_deterministic():
    set_global_seed(123)
    a_random = random.random()
    a_numpy = np.random.rand(5).tolist()

    set_global_seed(123)
    assert random.random() == a_random
    assert np.random.rand(5).tolist() == a_numpy


def test_set_global_seed_torch_deterministic():
    import torch

    set_global_seed(7)
    a = torch.randn(4)
    set_global_seed(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_generate_seed_in_range_and_varies():
    seeds = {generate_seed() for _ in range(50)}
    assert all(100_000 <= s <= 999_999 for s in seeds)
    # Vanishingly unlikely to collide to a single value across 50 draws.
    assert len(seeds) > 1


# ---------------------------------------------------------------------------
# Git capture
# ---------------------------------------------------------------------------

def test_capture_git_sha_in_real_repo():
    sha = capture_git_sha()
    # We're being run from inside the repo's pytest invocation.
    if sha is not None:
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_capture_git_sha_outside_repo(tmp_path):
    assert capture_git_sha(tmp_path) is None


def test_capture_git_dirty_outside_repo(tmp_path):
    assert capture_git_dirty(tmp_path) is None


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------

def test_hash_dataset_deterministic(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    h1 = hash_dataset([p])
    h2 = hash_dataset([p])
    assert h1 == h2
    assert len(h1) == 12


def test_hash_dataset_changes_with_content(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    h1 = hash_dataset([p])
    p.write_bytes(b"hello!")
    h2 = hash_dataset([p])
    assert h1 != h2


def test_hash_dataset_order_matters(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"BBBB")
    assert hash_dataset([a, b]) != hash_dataset([b, a])


def test_hash_dataset_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        hash_dataset([tmp_path / "nope.parquet"])


# ---------------------------------------------------------------------------
# save_run_config
# ---------------------------------------------------------------------------

def test_save_run_config_creates_dir_and_writes(tmp_path: Path):
    out = save_run_config(tmp_path / "deep" / "nested", seed=42)
    cfg = load_run_config(out)
    assert cfg["seed"] == 42
    assert "timestamp" in cfg
    assert "cwd" in cfg
    # git_sha and git_dirty are always present (may be None outside a repo).
    assert "git_sha" in cfg
    assert "git_dirty" in cfg


def test_save_run_config_with_hyperparams_and_extra(tmp_path: Path):
    out = save_run_config(
        tmp_path,
        seed=1,
        hyperparams={"lr": 1e-4, "gamma": 0.99},
        extra={"session_type": "PI+VC", "subject": "CM005"},
    )
    cfg = load_run_config(out)
    assert cfg["hyperparams"]["lr"] == 1e-4
    assert cfg["session_type"] == "PI+VC"
    assert cfg["subject"] == "CM005"


def test_save_run_config_auto_dataset_hash(tmp_path: Path):
    data_path = tmp_path / "fake_dataset.parquet"
    data_path.write_bytes(b"\x00\x01\x02fake_parquet_bytes\x03")
    out = save_run_config(tmp_path / "out", seed=1, dataset_paths=[data_path])
    cfg = load_run_config(out)
    assert "dataset_hash" in cfg
    assert len(cfg["dataset_hash"]) == 12


def test_save_run_config_explicit_dataset_hash_wins(tmp_path: Path):
    data_path = tmp_path / "fake.bin"
    data_path.write_bytes(b"x")
    out = save_run_config(
        tmp_path / "out",
        seed=1,
        dataset_paths=[data_path],
        extra={"dataset_hash": "manual123abc"},
    )
    cfg = load_run_config(out)
    assert cfg["dataset_hash"] == "manual123abc"


def test_save_run_config_json_is_well_formed(tmp_path: Path):
    out = save_run_config(tmp_path, seed=1, hyperparams={"x": 1})
    # Must round-trip through json.
    raw = Path(out).read_text()
    cfg = json.loads(raw)
    assert cfg["seed"] == 1
