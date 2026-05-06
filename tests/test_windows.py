"""Tests for corner_maze_rl.data.windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from corner_maze_rl.data.windows import (
    build_dt_dataset,
    build_windows_for_session,
    encode_session,
)


class _IdentityEncoder:
    """Encodes to (x, y, d, padding...) with output_dim=4 for easy assertions."""
    output_dim = 4

    def encode(self, x, y, direction, layout=None):
        return np.array([x, y, direction, 1.0], dtype=np.float32)


def _toy_session(n=10, session_id=1):
    return pd.DataFrame({
        "session_id": session_id,
        "step": list(range(n)),
        "action": [i % 5 for i in range(n)],
        "grid_x": [(i % 11) + 1 for i in range(n)],
        "grid_y": [((i + 3) % 11) + 1 for i in range(n)],
        "direction": [i % 4 for i in range(n)],
        "rewarded": [0] * n,
        "reward": [-0.001] * n,
        "return_to_go": [-0.001 * (n - i) for i in range(n)],
    })


# ---------------------------------------------------------------------------
# encode_session
# ---------------------------------------------------------------------------

def test_encode_session_shapes():
    df = _toy_session(n=7)
    arr = encode_session(df, _IdentityEncoder())
    assert arr["state"].shape == (7, 4)
    assert arr["pos"].shape == (7, 4)
    assert arr["rtg"].shape == (7, 1)
    assert arr["action"].shape == (7, 5)


def test_encode_session_one_hot_actions():
    df = _toy_session(n=5)
    arr = encode_session(df, _IdentityEncoder())
    # Actions 0..4 → identity matrix
    expected = np.eye(5, dtype=np.float32)
    np.testing.assert_array_equal(arr["action"], expected)


# ---------------------------------------------------------------------------
# build_windows_for_session
# ---------------------------------------------------------------------------

def test_build_windows_pads_and_slices_correctly():
    df = _toy_session(n=5)
    arr = encode_session(df, _IdentityEncoder())
    windows = build_windows_for_session(arr, context_size=3)

    # n=5 sessions → n=5 windows
    assert windows["state"].shape == (5, 3, 4)

    # Window 0: padded with 2 zeros, then real step 0
    assert np.all(windows["state"][0, :2, :] == 0.0)
    np.testing.assert_array_equal(windows["state"][0, 2, :], arr["state"][0])

    # Window 4 (last): real steps 2, 3, 4
    np.testing.assert_array_equal(windows["state"][4, 0, :], arr["state"][2])
    np.testing.assert_array_equal(windows["state"][4, 2, :], arr["state"][4])


# ---------------------------------------------------------------------------
# build_dt_dataset (end-to-end)
# ---------------------------------------------------------------------------

def test_build_dt_dataset_concatenates_sessions():
    df1 = _toy_session(n=4, session_id=1)
    df2 = _toy_session(n=6, session_id=2)
    big = pd.concat([df1, df2], ignore_index=True)
    ds = build_dt_dataset(big, _IdentityEncoder(), context_size=3)
    # Total windows = 4 + 6 = 10
    assert len(ds) == 10
    rtg, state, action, pos = ds[0]
    assert rtg.shape == (3, 1)
    assert state.shape == (3, 4)
    assert action.shape == (3, 5)
    assert pos.shape == (3, 4)
    assert isinstance(rtg, torch.Tensor)


def test_build_dt_dataset_rejects_empty():
    import pytest
    df = pd.DataFrame(columns=["session_id", "action", "grid_x", "grid_y",
                               "direction", "return_to_go"])
    with pytest.raises(ValueError):
        build_dt_dataset(df, _IdentityEncoder())


def test_build_dt_dataset_session_filter():
    df1 = _toy_session(n=4, session_id=1)
    df2 = _toy_session(n=6, session_id=2)
    big = pd.concat([df1, df2], ignore_index=True)
    ds_only_2 = build_dt_dataset(big, _IdentityEncoder(),
                                   context_size=3, session_filter=[2])
    assert len(ds_only_2) == 6
