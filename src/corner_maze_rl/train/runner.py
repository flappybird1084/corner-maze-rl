"""Model-agnostic session-sequence runner.

Adapted from legacy ``src/rl/session_runner.py`` with these changes for
the new repo:

  * Imports ``set_global_seed`` from the new ``utils.run_io`` module
    (legacy used a flat ``from seed_utils import``).
  * Wires the kill-switch (``train.kill_switch``) into the per-session
    loop so training halts on flat / dead curves.
  * Writes ``run_config.json``, ``killed_at.json`` and ``curves.parquet``
    using the per-run output schema described in plan §7.1.

The runner is *agnostic* to the model: callers pass four callables —
``make_env``, ``train_fn``, ``frozen_fn``, ``save_fn`` — that encapsulate
all model-specific behaviour. See ``md/dt-repo-plan.md`` §6 for the
``TrainableAgent`` protocol the callables conform to.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from corner_maze_rl.train.kill_switch import (
    DEFAULT_CONFIG,
    Decision,
    KillSwitchConfig,
    decide,
    killed_at_payload,
)
from corner_maze_rl.utils.run_io import save_run_config, set_global_seed


# ---------------------------------------------------------------------------
# Data serialization (mirrors legacy session_runner.save_episode_dataframe)
# ---------------------------------------------------------------------------

_NESTED_COLUMNS: tuple[str, ...] = (
    "trajectory", "trial_scores", "turn_scores", "session_scores",
    "trial_tags", "trial_configs", "sequence_labels",
)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} not JSON-serializable")


def save_episode_dataframe(df: pd.DataFrame, parquet_path: str | os.PathLike) -> None:
    """Persist an episode-rows DataFrame to parquet, JSON-encoding nested cols.

    pyarrow can't serialize mixed-type nested lists; the listed columns
    therefore get JSON-stringified. Round-trip with ``json.loads`` on read.
    """
    if df.empty:
        return
    df = df.copy()
    for col in _NESTED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, default=_json_default))
    df.to_parquet(parquet_path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_data(env_raw, all_episode_data: list[dict], episode_offset: int) -> int:
    """Append rows from env_raw.episode_data_rows with global episode numbering."""
    rows = getattr(env_raw, "episode_data_rows", None) or []
    for row in rows:
        row = dict(row)
        row["episode"] = row.get("episode", 0) + episode_offset
        all_episode_data.append(row)
    return episode_offset + len(rows)


def _per_session_score(env_raw) -> int | float:
    """Pull the per-session ``perfect_trial_count``-equivalent from env state.

    Falls back to total return if the env doesn't expose a trial-score field.
    """
    rows = getattr(env_raw, "episode_data_rows", None) or []
    if rows and "perfect_trial_count" in rows[-1]:
        return rows[-1]["perfect_trial_count"]
    if rows and "trial_scores" in rows[-1]:
        scores = rows[-1]["trial_scores"]
        if isinstance(scores, list):
            return int(sum(s for s in scores if s))
    return getattr(env_raw, "session_reward", 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    decision: Decision = Decision.CONTINUE
    n_sessions_run: int = 0
    scores: list[float] = field(default_factory=list)
    killed_at: dict | None = None
    df: pd.DataFrame | None = None


def run_session_sequence(
    session_types: str | Sequence[str],
    make_env: Callable[[str], tuple[Any, Any]],
    train_fn: Callable[[Any], None],
    frozen_fn: Callable[[Any], None],
    save_fn: Callable[[str], None],
    *,
    save_data_path: str | os.PathLike,
    model_save_dir: str | os.PathLike,
    seed: int | None = None,
    kill_switch_cfg: KillSwitchConfig | None = None,
    run_dir: str | os.PathLike | None = None,
    run_config_extra: dict | None = None,
) -> SessionResult:
    """Run an acquisition→probe sequence with kill-switch monitoring.

    Parameters mirror the legacy signature plus three additions:
      * ``kill_switch_cfg`` — overrides for the early-termination thresholds.
        Default = ``DEFAULT_CONFIG`` from ``train.kill_switch``.
      * ``run_dir`` — if provided, ``run_config.json`` / ``killed_at.json`` /
        ``curves.parquet`` are written there per plan §7.1.
      * ``run_config_extra`` — extra keys merged into ``run_config.json``
        (subject, session_type, encoder_list, ...).

    The kill switch evaluates after *every* session (acquisition or probe).
    On KILL_* it stops the loop and writes ``killed_at.json``.
    """
    if isinstance(session_types, str):
        session_types = [session_types]

    cfg = kill_switch_cfg or DEFAULT_CONFIG

    if seed is not None:
        set_global_seed(seed)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(
            run_dir,
            seed=seed if seed is not None else -1,
            extra={"session_types": list(session_types), **(run_config_extra or {})},
        )

    all_episode_data: list[dict] = []
    episode_offset = 0
    scores: list[float] = []
    decision = Decision.CONTINUE
    killed_payload: dict | None = None
    current_env_raw = None

    try:
        for session_type in session_types:
            is_acquisition = "acquisition" in session_type.lower()

            if is_acquisition:
                env_wrapped, env_raw = make_env(session_type)
                current_env_raw = env_raw
                if hasattr(env_raw, "policy_mode"):
                    env_raw.policy_mode = "updating"
                train_fn(env_wrapped)
                episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)

                sc = float(_per_session_score(env_raw))
                scores.append(sc)
                save_fn(str(Path(model_save_dir) / "model_post_acquisition"))
                current_env_raw = None
            else:
                # probe: frozen pass
                env_wrapped, env_raw = make_env(session_type)
                current_env_raw = env_raw
                if hasattr(env_raw, "policy_mode"):
                    env_raw.policy_mode = "frozen"
                frozen_fn(env_wrapped)
                episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)
                current_env_raw = None

                # probe: updating pass
                env_wrapped, env_raw = make_env(session_type)
                current_env_raw = env_raw
                if hasattr(env_raw, "policy_mode"):
                    env_raw.policy_mode = "updating"
                train_fn(env_wrapped)
                episode_offset = _collect_data(env_raw, all_episode_data, episode_offset)

                sc = float(_per_session_score(env_raw))
                scores.append(sc)
                current_env_raw = None

            # Kill-switch evaluation
            ks = decide(scores, cfg)
            decision = ks.decision
            if decision.is_terminal:
                if decision.is_kill:
                    killed_payload = killed_at_payload(ks, scores)
                    if run_dir is not None:
                        with open(run_dir / "killed_at.json", "w") as f:
                            json.dump(killed_payload, f, indent=2)
                break

        # Post-probes checkpoint (if loop ran any probes)
        if any("acquisition" not in st.lower() for st in session_types):
            save_fn(str(Path(model_save_dir) / "model_post_probes"))

    except KeyboardInterrupt:
        if current_env_raw is not None:
            episode_offset = _collect_data(current_env_raw, all_episode_data, episode_offset)
        df = pd.DataFrame(all_episode_data)
        save_episode_dataframe(df, save_data_path)
        raise

    df = pd.DataFrame(all_episode_data)
    save_episode_dataframe(df, save_data_path)

    if run_dir is not None and scores:
        pd.DataFrame({"session_index": range(len(scores)), "score": scores}) \
            .to_parquet(run_dir / "curves.parquet", engine="pyarrow")

    return SessionResult(
        decision=decision,
        n_sessions_run=len(scores),
        scores=scores,
        killed_at=killed_payload,
        df=df,
    )
