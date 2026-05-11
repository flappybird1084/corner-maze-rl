"""Tests for the DT trainer script."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_dt.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_dt", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load()


def _make_synthetic_dataset(tmp_path: Path, n_sessions: int = 3,
                            steps_per_session: int = 80) -> Path:
    """Build a fake actions_*.parquet shaped like the canonical schema.

    Columns: session_id, step, action, grid_x, grid_y, direction,
             rewarded, actions_to_reward.
    """
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(n_sessions):
        n = steps_per_session
        # plant two rewarded steps so actions_to_reward isn't trivial
        rewarded = np.zeros(n, dtype=np.int8)
        rewarded[n // 3] = 1
        rewarded[2 * n // 3] = 1
        atr = np.zeros(n, dtype=np.int32)
        countdown = 0
        for t in range(n - 1, -1, -1):
            if rewarded[t] == 1:
                countdown = 0
            else:
                countdown += 1
            atr[t] = countdown
        for step in range(n):
            rows.append({
                "session_id": sid,
                "step": step,
                "action": int(rng.integers(0, 5)),
                "grid_x": int(rng.integers(1, 12)),
                "grid_y": int(rng.integers(1, 12)),
                "direction": int(rng.integers(0, 4)),
                "rewarded": int(rewarded[step]),
                "actions_to_reward": int(atr[step]),
            })
    df = pd.DataFrame(rows)
    out = tmp_path / "actions.parquet"
    df.to_parquet(out, engine="pyarrow")
    return out


# --- Windowing helpers ---

def test_one_hot_shape_and_values(trainer):
    a = np.array([0, 1, 2, 3, 4, 4])
    oh = trainer._one_hot(a)
    assert oh.shape == (6, 5) and oh.dtype == np.float32
    assert oh.argmax(-1).tolist() == [0, 1, 2, 3, 4, 4]


def test_build_windows_front_pads_with_zeros(trainer):
    arrays = {"state": np.arange(10).reshape(10, 1).astype(np.float32),
              "rtg":   np.arange(10).reshape(10, 1).astype(np.float32),
              "action": np.eye(5, dtype=np.float32)[np.zeros(10, dtype=np.int64)],
              "pos":   np.arange(10).reshape(10, 1).astype(np.float32)}
    w = trainer.build_windows(arrays, context_size=4)
    # first window ends at step 0 → 3 zeros + step 0 (which is value 0.0)
    assert w["state"].shape == (10, 4, 1)
    assert w["state"][0, :3, 0].tolist() == [0, 0, 0]
    # last window ends at step 9 → values 6,7,8,9
    assert w["state"][9, :, 0].tolist() == [6, 7, 8, 9]


# --- Trainer end-to-end on synthetic data ---

def test_train_runs_one_epoch_softmax(trainer, tmp_path):
    data_path = _make_synthetic_dataset(tmp_path, n_sessions=2,
                                        steps_per_session=80)
    out_dir = tmp_path / "run"
    summary = trainer.train(
        data_path=data_path, out_dir=out_dir,
        arch="softmax",
        context_size=8, embed_dim=60, num_heads=2, num_layers=1,
        pos_encoding="learned",
        lr=1e-3, weight_decay=1e-4, batch_size=8,
        epochs=1, val_frac=0.1, seed=0, device="cpu",
        max_sessions=None,
    )
    assert summary["epochs_done"] == 1
    for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
        v = summary["last_epoch"][k]
        assert v == v and 0.0 <= v < 100.0
    assert (out_dir / "model.pt").is_file()
    assert (out_dir / "run_config.json").is_file()
    cfg = json.loads((out_dir / "run_config.json").read_text())
    assert cfg["arch"] == "softmax"
    assert cfg["state_dim"] == 60


def test_train_runs_softmax_decoupled(trainer, tmp_path):
    data_path = _make_synthetic_dataset(tmp_path, n_sessions=2,
                                        steps_per_session=80)
    out_dir = tmp_path / "run"
    summary = trainer.train(
        data_path=data_path, out_dir=out_dir,
        arch="softmax_decoupled",
        context_size=8, embed_dim=128, num_heads=2, num_layers=1,
        pos_encoding="learned",
        lr=1e-3, weight_decay=1e-4, batch_size=8,
        epochs=1, val_frac=0.1, seed=0, device="cpu",
        max_sessions=None,
    )
    assert summary["arch"] == "softmax_decoupled"
    assert summary["epochs_done"] == 1


def test_train_runs_linear_decoupled(trainer, tmp_path):
    data_path = _make_synthetic_dataset(tmp_path, n_sessions=2,
                                        steps_per_session=80)
    out_dir = tmp_path / "run"
    summary = trainer.train(
        data_path=data_path, out_dir=out_dir,
        arch="linear_decoupled",
        context_size=8, embed_dim=128, num_heads=2, num_layers=1,
        pos_encoding="learned",
        lr=1e-3, weight_decay=1e-4, batch_size=8,
        epochs=1, val_frac=0.1, seed=0, device="cpu",
        max_sessions=None,
    )
    assert summary["arch"] == "linear_decoupled"


def test_softmax_arch_rejects_state_dim_mismatch(trainer, tmp_path):
    """The coupled `softmax` arch should refuse state_dim != embed_dim."""
    data_path = _make_synthetic_dataset(tmp_path)
    with pytest.raises(ValueError, match="state_dim == embed_dim"):
        trainer.train(
            data_path=data_path, out_dir=tmp_path / "run",
            arch="softmax",
            context_size=8, embed_dim=128, num_heads=2, num_layers=1,
            pos_encoding="learned",
            lr=1e-3, weight_decay=1e-4, batch_size=8,
            epochs=1, val_frac=0.1, seed=0, device="cpu",
            max_sessions=None,
        )


def test_checkpoint_round_trips(trainer, tmp_path):
    from corner_maze_rl.models.decision_transformer import (
        DTConfig,
        DecisionTransformer,
    )
    data_path = _make_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "run"
    trainer.train(
        data_path=data_path, out_dir=out_dir,
        arch="softmax",
        context_size=8, embed_dim=60, num_heads=2, num_layers=1,
        pos_encoding="learned",
        lr=1e-3, weight_decay=1e-4, batch_size=8,
        epochs=1, val_frac=0.1, seed=0, device="cpu",
        max_sessions=None,
    )
    bundle = torch.load(out_dir / "model.pt", weights_only=False)
    cfg = DTConfig(**bundle["cfg"])
    model = DecisionTransformer(cfg)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    rtg = torch.zeros(2, cfg.context_size, 1)
    state = torch.zeros(2, cfg.context_size, cfg.embed_dim)
    action = torch.zeros(2, cfg.context_size, cfg.num_actions); action[:, :, 0] = 1.0
    with torch.no_grad():
        logits = model(rtg, state, action)
    assert logits.shape == (2, cfg.context_size, cfg.num_actions)


def test_max_sessions_limits_dataset(trainer, tmp_path):
    data_path = _make_synthetic_dataset(tmp_path, n_sessions=5,
                                        steps_per_session=40)
    out_dir = tmp_path / "run"
    trainer.train(
        data_path=data_path, out_dir=out_dir,
        arch="softmax",
        context_size=8, embed_dim=60, num_heads=2, num_layers=1,
        pos_encoding="learned",
        lr=1e-3, weight_decay=1e-4, batch_size=8,
        epochs=1, val_frac=0.1, seed=0, device="cpu",
        max_sessions=2,
    )
    cfg = json.loads((out_dir / "run_config.json").read_text())
    assert cfg["n_sessions"] == 2
