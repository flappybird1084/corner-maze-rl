# Yoked Dataset from Legacy Data

**Date:** 2026-05-10
**Branch / commits:** `yoked-from-legacy-data` — `be01f59` … `129f094`
**Scope:** data plumbing only. No model, env, or training-loop changes.

> **Status note.** This pipeline reaches an exposure-only subset of the
> dataset the [DT repo plan](dt-repo-plan.md) §16.2 envisions. Acquisition
> and probe sessions are **not** in the source we used and remain TODO.
> Treat the resulting `actions_with_returns.parquet` as a smoke-test
> dataset for the DT pipeline — useful for end-to-end wiring, not for
> the headline 2S2C-task experiment.

---

## 1. Why this exists

`corner-maze-rl` was designed to consume a *3-table normalized yoked
dataset*:

```
data/yoked/dataset/
├── subjects.parquet                  # one row per rat
├── sessions.parquet                  # one row per session
├── actions_synthetic_pretrial.parquet  # one row per env step
└── actions_real_pretrial.parquet
```

`scripts/setup_data.sh` was supposed to copy these from a sibling
checkout of [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy)
under `2S2C_task/` and `data/yoked/dataset/`. When we tried it:

- The legacy repo (cloned to `../corner-maze-rl-legacy`) **does not
  contain** `subjects.parquet` / `sessions.parquet` / `actions_*.parquet`.
- Its actual session data (`data/dataframes/all_sessions.parquet`) is
  **gitignored** (`.gitignore` line 5).
- It also lacks the `2S2C_task/embeddings/` and `yoking/` directories
  the plan references.

The 3-table form is the *output* of a yoking pipeline that was never
published. Until the canonical dataset is produced and shipped, we
build a smaller, exposure-only version from a sibling repository that
**does** ship per-rat session data: [`../rat-neural-model-2`](../rat-neural-model-2).

This document describes that conversion: what the source ships, the
schema mismatch, the conversion strategy, the resulting dataset's
limitations, and the test coverage.

---

## 2. Source dataset (`rat-neural-model-2`)

### 2.1 On-disk layout

```
rat-neural-model-2/data/parquet/sessions/
├── CM001/CM001_pre2.parquet
├── CM003/{CM003_pre1.parquet, CM003_pre2.parquet}
├── ...
└── CM064/CM064_pre1.parquet
```

39 rats (CM001..CM064 with gaps) × {`pre1`, `pre2`} files = **70 files**
total. 17 of those 70 are 0-row "stub" parquets with the right schema
but no rows; we silently skip them.

### 2.2 Per-row schema

```
columns: [filestem, X, Y, D, A, B, R]
dtypes : float64 throughout
```

Field semantics (recovered by inspection + cross-reference with
`rat-neural-model-2/notebooks/DTtrainer.ipynb`):

| col | meaning | corner-maze-rl mapping |
|---|---|---|
| `filestem` | maze layout name (e.g. `expa_x_x_xx`, `expb_sxxx_x_xx`) | implicit in `session_type`; not propagated |
| `X`, `Y` | grid position (0–12) | `grid_x`, `grid_y` (cast `int32`) |
| `D` | direction (0=east, 1=south, 2=west, 3=north) | `direction` (cast `int32`) |
| `A` | raw action ∈ {0, 1, 2, 5, 6} | remapped via `ACTION_MAP` → env action ∈ {0..4} |
| `B` | original step counter (negative values mixed in for some rows) | **dropped** |
| `R` | monotone-decreasing scalar of unclear semantics; range varies wildly across files | **dropped** — `compute_returns.py` recomputes reward by env replay |

### 2.3 `pre1` vs `pre2`

The naming convention ties to behavioral phase, confirmed by the
`filestem` values inside each file:

| filename suffix | `filestem` values observed | corner-maze-rl `session_type` |
|---|---|---|
| `_pre1.parquet` | `expa_x_x_xx` only | `"exposure"` |
| `_pre2.parquet` | `expb_sxxx_x_xx`, `expb_sexx_x_xx`, `expb_senx_x_xx`, `expb_senw_x_xx`, `expb_x_x_xx` | `"exposure_b"` |

`expa` = open-foraging exposure with all four wells active. `expb` =
the progressive barrier-drop exposure described in
`md/environment-architecture.md`.

### 2.4 Action remap

The rat-neural-model-2 notebooks (`DTtrainer.ipynb` and
`placeCellDttrainer.ipynb`) document:

```python
ACTION_MAP = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
# 0:Left, 1:Right, 2:Forward, 3:EnterWell, 4:Pause
mapped = ACTION_MAP.get(raw, 4)  # default-to-pause for unknown codes
```

Identical mapping in our converter, identical `default = 4` fallback.
Why two different conventions exist (raw 5/6 vs. env 3/4) is a legacy
artifact — the rat-tracking pipeline emits raw codes that the env's
discrete action space doesn't natively share.

---

## 3. Target dataset (corner-maze-rl 3-table form)

### 3.1 Schemas the existing pipeline expects

From `src/corner_maze_rl/data/load.py` and
`src/corner_maze_rl/scripts/build_returns_dataset.py`:

**`subjects.parquet`** — one row per rat:

| col | dtype | notes |
|---|---|---|
| `subject_id` | int | primary key |
| `subject_name` | str | e.g. `"CM001"`, used for `F1_SUBJECT_NAMES` lookup |
| `training_group` | str | `"PI"` / `"VC"` / `"PI+VC"` — feeds `PARADIGM_MAP` |
| `cue_goal_orientation` | str | feeds `agent_cue_goal_orientation` env kwarg |

**`sessions.parquet`** — one row per session:

| col | dtype | notes |
|---|---|---|
| `session_id` | int | primary key |
| `subject_id` | int | foreign key |
| `session_number` | int | 1-based ordering for a rat's sessions |
| `session_type` | str | yoked-style label; feeds `PARADIGM_MAP` |
| `trial_configs` | str | JSON-encoded list of `[arm, cue, goal, tag]` per trial |
| `seed` | int | optional; controls env reset seed |

**`actions_real_pretrial.parquet`** (or `actions_synthetic_pretrial`) —
one row per env step:

| col | dtype | notes |
|---|---|---|
| `session_id` | int | foreign key |
| `step` | int64 | 0-based step within session |
| `action` | int64 | env action ∈ {0..4} |
| `grid_x`, `grid_y`, `direction` | int32 | rat's recorded pose |

### 3.2 What the converter populates

Built by `scripts/build_yoked_from_legacy_data.py`:

| field | population strategy | rationale |
|---|---|---|
| `subject_id` | `enumerate(sorted(rat_dirs))` | deterministic, stable across re-runs |
| `subject_name` | dir name (e.g. `"CM001"`) | matches `F1_SUBJECT_NAMES` membership |
| `training_group` | `"PI"` placeholder | exposure paradigm doesn't read it; see §4.2 |
| `cue_goal_orientation` | `"NS"` placeholder | exposure paradigm doesn't read it |
| `session_id` | running counter, surviving (non-empty) files only | env replay needs unique per-session IDs |
| `subject_id` | `subj_idx[rat]` | fk into subjects |
| `session_number` | `1` for `pre1`, `2` for `pre2` | matches the rat's behavioral phase order |
| `session_type` | `"exposure"` / `"exposure_b"` | matches both env literal and our new `PARADIGM_MAP` keys |
| `trial_configs` | `"[]"` (JSON empty list) | `_first_trial_goal` parses with `json.loads`, returns `None` on `data == []` |
| `seed` | `int(md5(f"{rat}\|{stype}").hexdigest()[:6], 16)` | deterministic, varies by input, fits 24 bits |
| `step` | `np.arange(n)` | per-session 0-based |
| `action` | `df["A"].map(ACTION_MAP).fillna(4).astype(int64)` | matches notebook convention |
| `grid_x/y/direction` | rename `X/Y/D`, cast `int32` | direct preservation |

Dropped: `filestem`, `B`, `R` (see §2.2).

---

## 4. Conversion strategy and limitations

### 4.1 Schema dispatch (`PARADIGM_MAP` patch)

`src/corner_maze_rl/data/session_types.py::PARADIGM_MAP` is the
`(training_group, yoked_session_type) → env_paradigm_string` table that
`build_returns_dataset.py` calls to construct the env factory for each
session. It originally only had acquisition/probe paradigms.

Without an entry, `map_session_to_env_kwargs(...)` returns `None` and
the session is silently skipped as "unmapped" → zero rows reach
`actions_with_returns.parquet`.

We add two rows:
```python
("PI", "exposure"):    "exposure",
("PI", "exposure_b"):  "exposure_b",
```

**Only `"PI"` rows.** The converter normalizes everyone to
`training_group="PI"` for exposure data (since real metadata is
unavailable from the source — see §4.2). Adding `("VC","exposure")`
would be over-broad and could silently mask future
schema-validation issues. A test (`test_vc_exposure_not_added`) guards
against accidental over-add.

### 4.2 Placeholder metadata (`training_group="PI"`, `cue_goal_orientation="NS"`)

These two fields exist for downstream paradigms that branch on rat
training group (e.g. `"PI"` rats run no-cue acquisition; `"VC"` rats
run rotate-cue acquisition). The exposure paradigm doesn't read them —
`gen_grid_configuration_sequence` for `'exposure'` returns
`[all_wells_layout]` regardless of orientation.

Real values are recoverable from older lab metadata (e.g.
`corner-maze-rl-legacy/data/dataframes/median_masters_all_sessions.parquet`)
but not from `rat-neural-model-2`. If non-exposure data is added later,
the placeholders **must** be replaced with real values before any
acquisition/probe session can be replayed (the env will pick the wrong
paradigm otherwise).

### 4.3 Empty source files

17 of 70 source files are 0-row "stub" parquets. Reasons unknown
(possibly aborted recordings, or rats that never completed `pre1`).
The converter logs each skip and emits no `subjects` row removal — the
rat is kept in `subjects.parquet` even if all their session files were
empty (so the index space is stable across re-runs).

### 4.4 F1-subject filter

`F1_SUBJECT_NAMES` (CM057, CM058, CM059, CM060, CM061, CM063, CM064) is
applied by `build_returns_dataset.py`, not by our converter. F1 rats
have only exposure data here, but the filter still removes them because
the design doc treats f1 as "not yet yoked" pending an extension to the
yoking pipeline. We let the existing filter run as-is — those rows end
up in our `actions_real_pretrial.parquet` but are dropped during env
replay. Downstream net effect: `actions_with_returns.parquet` covers
**32 non-f1 rats / 46 sessions / ~90k steps**.

### 4.5 Pose drift during env replay

The env is deterministic and applies its own action semantics
(see `_apply_action`). When we feed in the rat's recorded action stream,
the env's reconstructed `agent_pos` may diverge from the rat's recorded
`(X, Y, D)` — typically because:

- The rat was sampled at a coarser tick than the env step, so a
  rat-recorded "forward" sometimes corresponds to multiple env forwards.
- The rat's `A=6` (mapped to `pause`) sometimes meant "wait for trigger"
  rather than literal env-no-op; the env happily no-ops, but the rat's
  next pose may be offset.

This is **acceptable** for our use case: `compute_returns.py` only
reads `env.step()`'s emitted reward, not the env's pose. The DT later
trains on `(rtg, state-encoded-from-rat-pose, rat-action)` — the state
vector comes from the rat's recorded pose, so drift in env pose
doesn't contaminate the training signal.

### 4.6 Reward density

Spot-check on the produced `actions_with_returns.parquet`:

- 90,229 total rows across 46 sessions.
- 416 positive-reward steps (0.46% of total). Reward values:
  - `+1.0605` (matches `WELL_REWARD_SCR = 1.061`, modulo float32 drift) — 416 occurrences.
- Step costs: `-0.0005` (forward / pickup / pause) and `-0.001` (turns).
- RTG recurrence `rtg[t] = reward[t] + rtg[t+1]` holds with max error
  `0.0` across all (session, trial_idx) groups.
- Action distribution: forward 35.5%, turns 37.7%, pause 23.7%, pickup
  3.0% — plausible for rat exposure foraging.

---

## 5. Usage

### 5.1 One-time generate

```bash
# 1. Convert legacy per-session parquets to 3-table form
.venv/bin/python scripts/build_yoked_from_legacy_data.py \
    --legacy-data-root ../rat-neural-model-2 \
    --out data/yoked/dataset

# 2. Replay through env to attach reward + RTG
.venv/bin/python -m corner_maze_rl.scripts.build_returns_dataset \
    --dataset-dir data/yoked/dataset \
    --actions-variant real_pretrial \
    --out data/yoked/dataset/actions_with_returns.parquet
```

Step 1: ~1 second. Step 2: ~3 minutes (env replay across 52 sessions).

Both outputs land under `data/yoked/dataset/`, which is gitignored by
the repo's existing `.gitignore`.

### 5.2 Re-running

The converter is idempotent: same inputs → same `subject_id` /
`session_id` / `seed` (md5 of `"{rat}|{session_type}"`). Re-runs
overwrite the three parquets cleanly.

### 5.3 Loading into the existing DT data pipeline

```python
from corner_maze_rl.data.load import YokedPaths, load_subjects, load_sessions
from corner_maze_rl.data.windows import build_dt_dataset
import pandas as pd

paths = YokedPaths.from_dir("data/yoked/dataset", actions_variant="real_pretrial")
df = pd.read_parquet("data/yoked/dataset/actions_with_returns.parquet")
# encoder = ... (instantiate per encoders/grid_cells.py etc.)
# dataset = build_dt_dataset(df, encoder, context_size=64)
```

The DT training loop itself remains TODO — the input pipeline is now
ready, the loop is roughly 50 lines mirroring
`rat-neural-model-2/notebooks/DTtrainer.ipynb` cell 2.

---

## 6. Test coverage

Two new test modules; 27 tests total; **0 regressions** in the
pre-existing 108 tests (full suite = 135 passed).

### 6.1 `tests/test_session_types.py` (6 tests)

- **Direct dict access** — both new keys present and resolve to the
  expected env paradigm strings.
- **`map_session_to_env_kwargs`** — full kwargs dict for both
  `exposure` and `exposure_b`.
- **Negative case** — `("VC", "exposure")` stayed unmapped (we only
  added `"PI"` rows).
- **Sanity** — pre-existing entry `("PI+VC", "Rotate Train")` still
  resolves.

### 6.2 `tests/test_build_yoked_from_legacy_data.py` (21 tests)

The converter lives outside the importable package
(`scripts/build_yoked_from_legacy_data.py`), so the test module loads
it via `importlib.util.spec_from_file_location(...)` and exposes it as
the `conv` fixture.

Coverage by area:

| area | tests |
|---|---|
| `ACTION_MAP` | known values, default-to-pause for `99` and `NaN` |
| `session_type_for(...)` | `_pre1` → `"exposure"`, `_pre2` → `"exposure_b"` |
| `deterministic_seed(...)` | stable, varies by input, fits 24-bit int |
| `discover(...)` | finds `pre1`+`pre2`, ignores `pre3` / `summary.parquet` / `notes.txt`, raises `FileNotFoundError` on missing root |
| `build_subjects(...)` | schema, ID enumeration, placeholder values |
| `build_sessions_and_actions(...)` | pre1/pre2 → exposure/exposure_b, action remap, legacy `B/R/filestem` dropped, empty sessions skipped, running `session_id`, `step` starts at 0, `trial_configs` round-trips through `json.loads` as `[]` |
| `build(...)` end-to-end | writes 3 parquets with correct shapes; `RuntimeError` on empty discovery |

### 6.3 Running

```bash
.venv/bin/python -m pytest tests/test_session_types.py \
                          tests/test_build_yoked_from_legacy_data.py -v
# 27 passed in ~1s
```

---

## 7. Future work

1. **Acquisition/probe data.** Source it from wherever the canonical
   yoking pipeline lives (likely on lab infrastructure, not GitHub),
   produce real `subjects.parquet` / `sessions.parquet` /
   `actions_synthetic_pretrial.parquet` / `actions_real_pretrial.parquet`
   alongside the existing exposure parquets. After that, drop the
   placeholder `training_group="PI"` and emit real values.
2. **Replace placeholder metadata.** When non-exposure data lands,
   `cue_goal_orientation` becomes load-bearing (drives env layout
   rotations); `training_group` drives `PARADIGM_MAP` dispatch.
3. **DT training loop.** ~50 lines following
   `rat-neural-model-2/notebooks/DTtrainer.ipynb` cell 2: AdamW,
   cross-entropy on action logits, optional cosine LR schedule. The
   data pipeline this doc describes is the only blocker.
4. **Retire the converter.** Once the canonical 3-table dataset ships
   from the legacy yoking pipeline, `scripts/setup_data.sh` can run
   verbatim and the converter (and this doc) can be deleted.

---

## 8. Files touched / produced

### Code (committed on branch `yoked-from-legacy-data`)

| commit | files | scope |
|---|---|---|
| `be01f59` | `src/corner_maze_rl/data/session_types.py` (+5), `scripts/build_yoked_from_rnm2.py` (new, 142) | initial converter + dispatch patch |
| `17b2b64` | path rename `build_yoked_from_rnm2.py` → `build_yoked_from_legacy_data.py` | rename (path only) |
| `4e6fab5` | content of converter | rename (content): `RNM2`→`LEGACY_DATA`, `rnm2_root`→`legacy_data_root`, `--rnm2-root`→`--legacy-data-root` |
| `129f094` | `tests/test_session_types.py`, `tests/test_build_yoked_from_legacy_data.py` | 27 tests |
| `31158ca` | `pyproject.toml` (+1) | declare `ipywidgets>=8.1.8` (used by the manual-control notebook) |

### Generated artifacts (gitignored, on disk under `data/yoked/`)

- `subjects.parquet` (39 rows)
- `sessions.parquet` (52 rows)
- `actions_real_pretrial.parquet` (157,552 rows)
- `actions_with_returns.parquet` (90,229 rows; produced by
  `build_returns_dataset.py` after the converter)

### Untouched

- `src/corner_maze_rl/data/{compute_returns,windows,load}.py`
- `src/corner_maze_rl/scripts/build_returns_dataset.py`
- `src/corner_maze_rl/env/`, `models/`, `encoders/`
- All other tests in `tests/`

---

## 9. References

- [md/dt-repo-plan.md](dt-repo-plan.md) §16 — original data-source manifest.
- [md/environment-architecture.md](environment-architecture.md) — env paradigms, exposure structure.
- [md/sr-yoked-negative-results.md](sr-yoked-negative-results.md) — earlier negative result on yoked data; informs the placeholder-metadata caveat above.
- `rat-neural-model-2/notebooks/DTtrainer.ipynb` — source of the
  `ACTION_MAP` and the DT training-loop reference implementation.
