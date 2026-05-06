"""Context-window builder for Decision Transformer training.

Consumes the per-step ``actions_with_returns.parquet`` (output of
``compute_returns.py``) and produces fixed-length context windows of the
form ``(rtg, state, action, pos)`` per plan §6.2.

Differences from the legacy ``DTtrainer.ipynb`` cell 1 dataloader:

  * RTG is the **real** env-derived ``return_to_go`` column (per plan §4),
    not the fake ``1 - steps_to_go/500`` proxy. Pass ``rtg_col`` to choose.
  * Action remap is gone — the yoked dataset already uses 5-action ints
    (0..4) aligned with the env per plan §6.2.
  * Position vectors come from any registered encoder, not just the
    bundled grid_cells dict.

Schema of one window (all torch tensors, shape ``(K, *)`` with
``K = context_size``):

  * ``rtg``    — float32, ``(K, 1)``
  * ``state``  — float32, ``(K, encoder.output_dim)``
  * ``action`` — float32, ``(K, num_actions)`` one-hot
  * ``pos``    — float32, ``(K, encoder.output_dim)`` (same as state for
                grid-cell pose encoding; available for the optional
                "spatial" positional-encoding mode)

Front-padded with ``context_size - 1`` zero tokens so the first real
step has a full context. Each session contributes ``len(session)`` windows.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def _one_hot(actions: np.ndarray, num_actions: int) -> np.ndarray:
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions] = 1.0
    return out


def encode_session(
    session_df: pd.DataFrame,
    encoder,
    *,
    rtg_col: str = "return_to_go",
    action_col: str = "action",
    x_col: str = "grid_x",
    y_col: str = "grid_y",
    dir_col: str = "direction",
    num_actions: int = 5,
) -> dict[str, np.ndarray]:
    """Encode one session into per-step state/action/RTG arrays.

    Returns a dict with keys ``state``, ``action``, ``rtg``, ``pos`` —
    each as a numpy array. State and pos are produced by the encoder;
    they're identical when using a single positional encoder, so the
    caller can choose to drop one.
    """
    n = len(session_df)
    state = np.zeros((n, encoder.output_dim), dtype=np.float32)
    pos = np.zeros((n, encoder.output_dim), dtype=np.float32)
    xs = session_df[x_col].to_numpy(dtype=np.int32)
    ys = session_df[y_col].to_numpy(dtype=np.int32)
    ds = session_df[dir_col].to_numpy(dtype=np.int32)
    for i, (x, y, d) in enumerate(zip(xs, ys, ds)):
        v = encoder.encode(int(x), int(y), int(d))
        state[i] = v
        pos[i] = v

    rtg = session_df[rtg_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    actions = session_df[action_col].to_numpy(dtype=np.int64)
    actions = np.clip(actions, 0, num_actions - 1)
    action_oh = _one_hot(actions, num_actions)
    return {"state": state, "action": action_oh, "rtg": rtg, "pos": pos}


def build_windows_for_session(
    session_arrays: dict[str, np.ndarray],
    *,
    context_size: int,
) -> dict[str, np.ndarray]:
    """Slice a session's per-step arrays into fixed-length windows.

    Front-pads with ``context_size - 1`` zeros so window i = steps
    ``[i, i+context_size)`` after padding, i.e. ends at real step i.
    """
    pad = context_size - 1
    n = len(session_arrays["state"])

    def pad_front(arr: np.ndarray) -> np.ndarray:
        if pad == 0:
            return arr
        zeros = np.zeros((pad, *arr.shape[1:]), dtype=arr.dtype)
        return np.concatenate([zeros, arr], axis=0)

    padded = {k: pad_front(v) for k, v in session_arrays.items()}

    out = {k: np.empty((n, context_size, *v.shape[1:]), dtype=v.dtype)
           for k, v in session_arrays.items()}
    for i in range(n):
        for k in session_arrays:
            out[k][i] = padded[k][i: i + context_size]
    return out


def build_dt_dataset(
    actions_with_returns: pd.DataFrame,
    encoder,
    *,
    context_size: int = 64,
    session_col: str = "session_id",
    num_actions: int = 5,
    rtg_col: str = "return_to_go",
    progress: bool = False,
    session_filter: Iterable[int] | None = None,
) -> TensorDataset:
    """Build a TensorDataset from a per-step actions_with_returns dataframe.

    Iterates over sessions in *actions_with_returns*, encodes each into
    ``(rtg, state, action, pos)`` arrays, slices into context windows,
    concatenates across sessions, and returns a TensorDataset whose tuple
    order is **(rtg, state, action, pos)** — matching the legacy DTtrainer
    convention so DT model code can ingest it directly.

    Parameters
    ----------
    actions_with_returns : pd.DataFrame
        Per-step DataFrame produced by ``build_returns_dataset.py``.
        Must include: session_col, action, grid_x, grid_y, direction,
        return_to_go (or whatever ``rtg_col`` you pick).
    encoder : StateEncoder
        Provides ``encode(x, y, direction)`` → vector of length
        ``encoder.output_dim`` and that ``output_dim`` attribute.
    context_size : int
        Tokens per RSA stream (legacy default 64).
    session_filter : iterable of session ids, optional
        Restrict to a subset of sessions (e.g. for tune/report split).
    """
    sessions: list[int]
    if session_filter is not None:
        sessions = list(session_filter)
    else:
        sessions = list(actions_with_returns[session_col].unique())

    iterator = sessions
    if progress:
        from tqdm import tqdm
        iterator = tqdm(sessions, desc="windowing sessions")

    accum: dict[str, list[np.ndarray]] = {"rtg": [], "state": [], "action": [], "pos": []}
    for sid in iterator:
        sess_df = actions_with_returns[actions_with_returns[session_col] == sid]
        if len(sess_df) == 0:
            continue
        arrays = encode_session(sess_df, encoder, num_actions=num_actions, rtg_col=rtg_col)
        windows = build_windows_for_session(arrays, context_size=context_size)
        for k in accum:
            accum[k].append(windows[k])

    if not accum["state"]:
        raise ValueError("build_dt_dataset: no sessions produced any windows")

    rtg = torch.from_numpy(np.concatenate(accum["rtg"], axis=0))
    state = torch.from_numpy(np.concatenate(accum["state"], axis=0))
    action = torch.from_numpy(np.concatenate(accum["action"], axis=0))
    pos = torch.from_numpy(np.concatenate(accum["pos"], axis=0))
    return TensorDataset(rtg, state, action, pos)
