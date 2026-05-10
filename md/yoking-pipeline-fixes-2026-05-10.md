# Yoking pipeline fixes — 2026-05-10

Eight delta fixes ported from `corner-maze-rl-legacy/yoking/map_to_minigrid_actions.py` into `corner_maze_rl.yoking.map_to_minigrid_actions`. All fixes preserve the **env grid is ground truth** principle: the yoking pipeline must produce action sequences that respect the env's barrier and trigger semantics at every step.

## Fixes (in order they appear in the file)

### 1. Connectivity-aware remap helpers
Added `_reachable_from(anchor, blocked)` (BFS over `WALKABLE_CELLS` with active barrier set) and `_remap_unreachable_to_reachable(grid_x, grid_y, blocked, anchor)` (push frames at *walkable-but-isolated* cells back to reachable cells). Catches frames in barrier-isolated pockets that `_remap_blocked_to_neighbor` doesn't address (cells that are walkable but cut off from `anchor` by the active barrier set).

### 2. Per-frame ITI co-sim
Added `_simulate_iti_barriers_per_frame(grid_x, grid_y, iti_configs, initial_idx, BL, TL)` and `_remap_with_per_frame_barriers(grid_x, grid_y, blocked_per_frame, anchor)`. The env flips `_iti_config_idx` when the agent steps on an A/B trigger, so the active barrier set during an ITI is *not constant*. The co-sim walks the rat's frames, replicates the env's layout swap on trigger crossings, and remaps each frame against the barriers active at *that* frame.

### 3. Inner WE-LF else-branch corner-trick fallback
In `_generate_segment_actions`'s well-exit-L/F check: when `dest` doesn't directly match `left_dest[0]` or `fwd_dest[0]`, BFS to dest and check `path[1]`. If it matches the corner-trick destination, emit a single corner-trick action and let the regular bridge handle the rest. Handles tracker phantom jumps that skip past the corner-trick destination cell, e.g. `(10,2)→(10,6)` skipping `(10,3)`.

### 4. Goal-well-as-first-run synthesis
In the trial-continuation block: when `runs[0]` is itself the goal well (rat tracker placed rat in well at start of ITI-phase data, often because the trial-phase boundary closed at the well visit), the segment generator's well-visit detection won't fire — it requires a `(corner, well)` run pair. Synthesize the corner approach + well-visit block (PICKUP + RR/LL + F) and slice off the goal-well run so the rest can be processed as normal ITI.

### 5. Outer ITI corner-trick fallback in well-exit check
Same pattern as fix #3, but in the *outer* ITI block's well-exit-L/F check (before the segment generator runs). When `first_run_pos` is further than the corner-trick destination, BFS-check `path[1]` for the L/F-trick destination match.

### 6. Reordered ITI well-exit-L/F before bridge
The outer well-exit-L/F check used to fire *after* the bridge from `current_pos` to `first_run_pos`. With this ordering, when the tracker skipped the corner frame, the bridge would emit a regular `L+F` (turn-then-walk) — but the env collapses `L` at a `WELL_EXIT_LEFT` corner into one combined turn-and-step (corner-trick), creating a phantom `(corner, post-L-dir)` row. Moved the well-exit-L/F check *before* the bridge so the corner-trick wins.

### 7. Sequential `barrier_stage` advance in exposure_b
Replaced `if new_stage > barrier_stage` with `if new_stage == barrier_stage + 1` plus a 5-frame minimum on the run length. The env's exposure_b trigger system requires *sequential* zone-target crossings — visiting `(6,8)` while in stage 2 isn't a trigger event because the stage-5 trigger at `(6,8)` is only armed in stage 4. Without sequential enforcement, brief tracker phantoms at later-stage zone targets caused yoking to jump `barrier_stage` ahead of the env's actual stage, leading to action emits the env couldn't follow. Net effect: phantom flicker visits at late-stage zones now fall through to the regular reachable-set check instead of triggering a stage skip.

### 8. `iti_sim_configs` plumbing
Added the `iti_sim_configs` parameter to `generate_acquisition_actions` and the corresponding setup in `build_action_sequence`. Each entry is `(iti_configs_tuple3, initial_idx)` per ITI, threaded through to the per-frame co-sim.

## Validation

- **In-scope:** 422/422 sessions pass (48 subjects: CM000–CM064 PI/VC/PI+VC/PI+VC_f1).
- **Overall:** 687/688 (only `PG34_7` fails, out of scope, same failure as before fixes).
- **Smoke test:** `corner-maze-build-returns` succeeded with 284,573 rows in `actions_with_returns.parquet`. Skips match the existing paradigm-map skip list (Exposure variants + VC_DREADDs Rotate Train).

## Principle: env grid is ground truth

When validating yoked rat data against `CornerMazeEnv` (e.g. `check_divergence`), **the grid the env generates from `trial_configs` and the layout tables is authoritative**. Barriers, cues, wells, and triggers in that grid are the truth — the yoked action sequence must respect them, even when the rat's recorded coordinates appear to violate them. Tracker frames that map to env-defined barrier cells, or to walkable-but-barrier-isolated pockets, are zone-boundary mapping artifacts, not real positions. A divergence at a `Barrier()` cell is a yoking-side bug, not an env-side one — fix the yoking pipeline (remap, BFS bridge, filter), don't soften the env.

There is no per-session physical-barrier-state stream in the analysis db (no column in `trials.parquet` or `phases.parquet` describing which barriers were physically up). The env's `trl_*_*_*` and `iti_*_*_*` layouts encode the experimental design's barrier configuration.
