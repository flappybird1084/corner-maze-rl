"""Decision Transformer trainer.

Reads one of the canonical action parquets at ``data/yoked/dataset/``:

    actions_synthetic_pretrial.parquet   (Acquisition, synthetic pretrials — default)
    actions_real_pretrial.parquet        (Acquisition, real pretrials)
    actions_exposure.parquet             (Exposure only)

Builds ``(rtg, state, action)`` context windows per session, then trains a
``DecisionTransformer`` (or a sibling architecture via ``--arch``) with AdamW
and cross-entropy on the rat's action labels.

The RTG conditioning signal is the ``actions_to_reward`` integer column
(steps until the next reward in the session), cast to float and fed to
``embed_rtg``. The DT learns to map this conditioning + recent context →
action distribution.

Outputs (under ``runs/dt/<run_name>/``):
  * model.pt          torch.save of state_dict + cfg
  * metrics.jsonl     one record per epoch
  * train.log         stdout mirror
  * run_config.json   hyperparams + dataset stats
  * summary.json      first/last epoch + wall time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from corner_maze_rl.encoders.grid_cells import GridCellEncoder
from corner_maze_rl.models.decision_transformer import DTConfig, DecisionTransformer
from corner_maze_rl.models.decision_transformer_decoupled_dimension import (
    DTConfigDecoupled,
    DecisionTransformerDecoupled,
)
from corner_maze_rl.models.linear_decision_transformer import (
    LinearDTConfig,
    LinearDecisionTransformer,
)
from corner_maze_rl.models.linear_decision_transformer_decoupled_dimension import (
    LinearDTConfigDecoupled,
    LinearDecisionTransformerDecoupled,
)


NUM_ACTIONS = 5


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _one_hot(actions: np.ndarray, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions] = 1.0
    return out


def encode_session(session_df: pd.DataFrame, encoder) -> dict[str, np.ndarray]:
    """One session → ``(state, action, rtg, pos)`` per-step arrays.

    ``state`` and ``pos`` are identical (single encoder); the duplicate is
    only useful when the DT uses ``pos_encoding="spatial"``.
    """
    n = len(session_df)
    state = np.zeros((n, encoder.output_dim), dtype=np.float32)
    xs = session_df["grid_x"].to_numpy(dtype=np.int32)
    ys = session_df["grid_y"].to_numpy(dtype=np.int32)
    ds = session_df["direction"].to_numpy(dtype=np.int32)
    for i in range(n):
        state[i] = encoder.encode(int(xs[i]), int(ys[i]), int(ds[i]))
    rtg = session_df["actions_to_reward"].to_numpy(dtype=np.float32).reshape(-1, 1)
    actions = session_df["action"].to_numpy(dtype=np.int64).clip(0, NUM_ACTIONS - 1)
    action_oh = _one_hot(actions)
    return {"state": state, "action": action_oh, "rtg": rtg, "pos": state.copy()}


def build_windows(arrays: dict[str, np.ndarray], context_size: int) -> dict[str, np.ndarray]:
    """Front-pad with ``context_size - 1`` zeros so window i ends at step i."""
    pad = context_size - 1
    n = len(arrays["state"])

    def front_pad(arr: np.ndarray) -> np.ndarray:
        if pad == 0:
            return arr
        zeros = np.zeros((pad, *arr.shape[1:]), dtype=arr.dtype)
        return np.concatenate([zeros, arr], axis=0)

    padded = {k: front_pad(v) for k, v in arrays.items()}
    out = {k: np.empty((n, context_size, *v.shape[1:]), dtype=v.dtype)
           for k, v in arrays.items()}
    for i in range(n):
        for k in arrays:
            out[k][i] = padded[k][i: i + context_size]
    return out


def build_dt_dataset(df: pd.DataFrame, encoder, context_size: int) -> TensorDataset:
    """All sessions → one TensorDataset of (rtg, state, action, pos) tuples."""
    acc = {"rtg": [], "state": [], "action": [], "pos": []}
    for sid in df["session_id"].unique():
        sdf = df[df["session_id"] == sid].sort_values("step")
        if len(sdf) == 0:
            continue
        arrays = encode_session(sdf, encoder)
        windows = build_windows(arrays, context_size=context_size)
        for k in acc:
            acc[k].append(windows[k])
    if not acc["state"]:
        raise ValueError("no sessions produced any windows")
    rtg = torch.from_numpy(np.concatenate(acc["rtg"], axis=0))
    state = torch.from_numpy(np.concatenate(acc["state"], axis=0))
    action = torch.from_numpy(np.concatenate(acc["action"], axis=0))
    pos = torch.from_numpy(np.concatenate(acc["pos"], axis=0))
    return TensorDataset(rtg, state, action, pos)


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

ARCH_CHOICES = ("softmax", "linear", "softmax_decoupled", "linear_decoupled")


def build_model(
    arch: str,
    *,
    state_dim: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    context_size: int,
    pos_encoding: str,
):
    if arch == "softmax":
        if state_dim != embed_dim:
            raise ValueError(
                f"arch=softmax requires state_dim == embed_dim "
                f"(got {state_dim} vs {embed_dim}); use softmax_decoupled instead"
            )
        cfg = DTConfig(embed_dim=embed_dim, num_actions=NUM_ACTIONS,
                       context_size=context_size, num_heads=num_heads,
                       num_layers=num_layers, pos_encoding=pos_encoding)
        return DecisionTransformer(cfg), cfg
    if arch == "linear":
        if state_dim != embed_dim:
            raise ValueError(
                f"arch=linear requires state_dim == embed_dim "
                f"(got {state_dim} vs {embed_dim}); use linear_decoupled instead"
            )
        cfg = LinearDTConfig(embed_dim=embed_dim, num_actions=NUM_ACTIONS,
                             context_size=context_size, num_heads=num_heads,
                             num_layers=num_layers, pos_encoding=pos_encoding)
        return LinearDecisionTransformer(cfg), cfg
    if arch == "softmax_decoupled":
        cfg = DTConfigDecoupled(state_dim=state_dim, embed_dim=embed_dim,
                                num_actions=NUM_ACTIONS,
                                context_size=context_size, num_heads=num_heads,
                                num_layers=num_layers, pos_encoding=pos_encoding)
        return DecisionTransformerDecoupled(cfg), cfg
    if arch == "linear_decoupled":
        cfg = LinearDTConfigDecoupled(state_dim=state_dim, embed_dim=embed_dim,
                                      num_actions=NUM_ACTIONS,
                                      context_size=context_size, num_heads=num_heads,
                                      num_layers=num_layers, pos_encoding=pos_encoding)
        return LinearDecisionTransformerDecoupled(cfg), cfg
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCH_CHOICES}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def log(msg: str, log_path: Path) -> None:
    print(msg, flush=True)
    with log_path.open("a") as f:
        f.write(msg + "\n")


def train(
    *,
    data_path: Path,
    out_dir: Path,
    arch: str,
    context_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    pos_encoding: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    val_frac: float,
    seed: int,
    device: str,
    max_sessions: int | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"
    log_path.write_text("")
    metrics_path.write_text("")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log(f"[load] reading {data_path}", log_path)
    df = pd.read_parquet(data_path)
    if max_sessions is not None:
        keep = sorted(df["session_id"].unique())[:max_sessions]
        df = df[df["session_id"].isin(keep)].reset_index(drop=True)
    log(f"[load] {len(df):,} rows across {df['session_id'].nunique()} sessions",
        log_path)

    encoder = GridCellEncoder()
    state_dim = encoder.output_dim

    log(f"[windows] building (context_size={context_size}) ...", log_path)
    full = build_dt_dataset(df, encoder, context_size=context_size)
    n = len(full)
    log(f"[windows] {n:,} windows total  state_dim={state_dim}", log_path)

    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    log(f"[split] train={n_train:,}  val={n_val:,}  batch_size={batch_size}",
        log_path)

    model, cfg = build_model(
        arch,
        state_dim=state_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        context_size=context_size,
        pos_encoding=pos_encoding,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    d_head = embed_dim // num_heads
    log(
        f"[model] arch={arch} params={n_params:,} state_dim={state_dim} "
        f"embed_dim={embed_dim} d_head={d_head} pos_encoding={pos_encoding} "
        f"device={device}",
        log_path,
    )

    run_config = {
        "data_path": str(data_path),
        "n_rows": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_windows": n, "n_train": n_train, "n_val": n_val,
        "arch": arch,
        "state_dim": state_dim,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "context_size": context_size,
        "pos_encoding": pos_encoding,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "device": device,
        "n_params": n_params,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_correct = 0
        sum_tokens = 0
        for rtg, state, action, pos in train_loader:
            rtg = rtg.to(device)
            state = state.to(device)
            action = action.to(device)
            pos = pos.to(device) if pos_encoding == "spatial" else None

            logits = model(rtg, state, action, pos_vecs=pos)
            targets = action.argmax(dim=-1)
            loss = criterion(logits.reshape(-1, NUM_ACTIONS), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * targets.numel()
            sum_correct += (logits.argmax(dim=-1) == targets).sum().item()
            sum_tokens += targets.numel()

        train_loss = sum_loss / sum_tokens
        train_acc = sum_correct / sum_tokens

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_tokens = 0
        with torch.no_grad():
            for rtg, state, action, pos in val_loader:
                rtg = rtg.to(device)
                state = state.to(device)
                action = action.to(device)
                pos = pos.to(device) if pos_encoding == "spatial" else None
                logits = model(rtg, state, action, pos_vecs=pos)
                targets = action.argmax(dim=-1)
                loss = criterion(logits.reshape(-1, NUM_ACTIONS), targets.reshape(-1))
                v_loss += loss.item() * targets.numel()
                v_correct += (logits.argmax(dim=-1) == targets).sum().item()
                v_tokens += targets.numel()
        val_loss = v_loss / max(v_tokens, 1)
        val_acc = v_correct / max(v_tokens, 1)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": time.time() - t0,
        }
        history.append(rec)
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(
            f"[epoch {epoch:3d}] train_loss={train_loss:.4f} acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.3f} "
            f"| {rec['elapsed_sec']:.1f}s",
            log_path,
        )

    elapsed = time.time() - t0
    ckpt_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                "arch": arch}, ckpt_path)
    summary = {
        "arch": arch,
        "epochs_done": len(history),
        "first_epoch": history[0] if history else None,
        "last_epoch": history[-1] if history else None,
        "wall_time_sec": elapsed,
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[done] {elapsed:.1f}s  -> {ckpt_path}", log_path)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/yoked/dataset/actions_synthetic_pretrial.parquet")
    p.add_argument("--out", default="runs/dt/default")
    p.add_argument("--arch", default="softmax", choices=ARCH_CHOICES)
    p.add_argument("--context-size", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=60)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--pos-encoding", default="learned",
                   choices=["learned", "sinusoidal", "spatial", "none"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cuda / mps / cpu; auto-detect when omitted")
    p.add_argument("--max-sessions", type=int, default=None,
                   help="restrict to first N sessions (smoke runs)")
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    train(
        data_path=Path(args.data),
        out_dir=Path(args.out),
        arch=args.arch,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        pos_encoding=args.pos_encoding,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        max_sessions=args.max_sessions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
