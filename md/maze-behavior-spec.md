# 2S2C Maze Behavior Specification

Reference for the behavioral rules governing the two-start two-choice (2S2C) task.
Adapted from `corner-maze-analysis/docs/maze_behavior_spec.md` with MiniGrid environment mappings.

## Training groups

Three acquisition conditions, distinguished by what cues are available to localize the goal corner. The maze geometry and goal location are identical across groups; only the cue condition differs.

| Group | Cue condition | TrainType | Available strategies |
|-------|---------------|-----------|----------------------|
| **PI+VC** | Fixed visual cue (single landmark in a constant world position across trials) | `f` (fixed) | Path integration *and* visual cue |
| **PI** | No cue (dark room, no visible landmarks) | `n` (no cue) | Path integration only |
| **VC** | Rotating visual cue (landmark rotated trial-by-trial so its world position is uncorrelated with the goal) | `r` (rotating) | Visual cue only — path integration to a fixed world location is disrupted |

Goal corner is fixed in world coordinates for PI+VC and PI. For VC, the goal corner rotates with the cue so the cue→goal relationship is constant in cue-frame coordinates but the world-frame goal location varies.

Sex-balanced cohort sizes: PI+VC = 17 (8M, 9F), PI = 12 (6M, 6F), VC = 13 (7M, 6F). Manuscript Figs 2–5 use these 42 rats.

### Subject IDs

CM identifiers are the canonical names; numeric `subject_id` values are assigned by `subjects.parquet` and are **not** sequential (gaps from pilots and pre-criterion dropouts). Authoritative list: `corner-maze-analysis/config/paper_cohort.yaml`.

**PI+VC** (n=17): CM000, CM001, CM002, CM003, CM004, CM005, CM006, CM007, CM008, CM009, CM010, CM011, CM014, CM015, CM016, CM017, CM018

**PI** (n=12): CM023, CM024, CM027, CM036, CM037, CM046, CM049, CM050, CM051, CM052, CM053, CM054

**VC** (n=13): CM025, CM026, CM031, CM035, CM038, CM040, CM041, CM042, CM043, CM044, CM045, CM055, CM056

A separate **PI+VC_f1** cohort (n=6: CM057, CM058, CM060, CM061, CM063, CM064) appears in Supplementary Figure 1 only — same fixed-cue training as PI+VC but with a "twist" probe variant.

## Arm structure

The plus maze has four arms connecting the center intersection to the perimeter. Each arm spans two zones: an inner zone adjacent to the center and an outer zone toward the perimeter.

### Real maze (zone-based)

| Arm | Inner zone | Outer zone | Direction |
|-----|-----------|-----------|-----------|
| North | 12 | 13 | increasing x |
| South | 10 | 9 | decreasing x |
| East | 15 | 19 | increasing y |
| West | 7 | 3 | decreasing y |

The start arm for each trial is identified by the inner zone: North=12, South=10, East=15, West=7.

### MiniGrid mapping

| Arm | Grid coordinates (approx.) | Start pose (x, y, dir) |
|-----|---------------------------|----------------------|
| North | (6, 2)-(6, 5) | (6, 3, 0) facing East |
| East | (7, 6)-(10, 6) | (9, 6, 1) facing South |
| South | (6, 7)-(6, 10) | (6, 9, 2) facing West |
| West | (2, 6)-(5, 6) | (3, 6, 3) facing North |

Center intersection: (6, 6). Grid size: 13x13.

## Session structure

Each subject runs multiple sessions across days. Sessions are grouped into phases:

1. **Acquisition** — the rat learns to navigate to the correct goal corner. The goal corner is fixed within each session but can differ between sessions depending on the subject's cue-goal orientation.
2. **Probes** — novel route, no cue, rotation, reversal tests after acquisition criterion is met.

### Within each acquisition session

1. **Exposure A** — free exploration, all 4 wells rewarded via cycle-alternation (each well resets after all 4 visited).
2. **Exposure B** — timed exploration with barriers appearing partway through.
3. **Acquisition trials** — pretrial → trial → ITI cycling. Goal well fixed for the session.

**Important for RL:** Exposure phases should NOT be used for reward-weight (w) learning — all 4 wells give reward during exposure, which teaches the wrong reward mapping for trial-phase goal-directed behavior.

## Trial structure

Each training trial follows three phases:

1. **Pretrial** — agent is confined to the start arm (barriers closed). Minimum duration enforced (PRETRIAL_MIN_STEPS = 7 in MiniGrid). Cue is presented (visual cue on the wall indicating goal direction).
2. **Trial** — barriers open, agent navigates from start arm through two choice points to the goal corner. Ends when the correct well is entered.
3. **ITI** — return to a start arm for the next trial. Barriers reconfigure for open navigation. Sub-phases controlled by A/B triggers.

## Turn detection

Turns are defined from the agent's body frame as it moves through the maze. "Right" and "Left" are relative to the agent's heading direction.

### 1st turn — center intersection

The agent enters the center from its start arm and exits into one of the two perpendicular arms.

| Start arm | Right | Left |
|-----------|-------|------|
| North | N → Center → West | N → Center → East |
| South | S → Center → East | S → Center → West |
| East | E → Center → North | E → Center → South |
| West | W → Center → South | W → Center → North |

### 2nd turn — perimeter intersection

The agent traverses the perpendicular arm outward and enters a perimeter segment, turning toward one of two corners.

### Route encoding

The two turns concatenated: LL, LR, RL, or RR. From either start arm there is exactly one correct route to the goal corner.

## Well-visit detection

Goal wells are the four corner zones: {1 (SW), 5 (NW), 17 (SE), 21 (NE)}. Each trial has one rewarded well (`trigger_zone`) and three error wells.

### Error-well visit

Registered when both conditions hold:
- Continuously in the error zone for >= 10 ms (`pass_time_in_error_zone`)
- Time since last registered error visit >= 2 s (`pass_time_out_of_error_zone`)

The debounce prevents re-counting when the rat lingers or re-enters the same well.

### Reward-well entry

Registered when continuously in `trigger_zone` for >= 250 ms (`pass_time_reward_zone`). Terminates the trial.

## Goal well structure

| Corner | Real zone | MiniGrid well pos | MiniGrid corner pos |
|--------|----------|-------------------|-------------------|
| SW | 1 | (1, 1) | (2, 2) |
| NW | 5 | (1, 11) | (2, 10) |
| SE | 17 | (11, 1) | (10, 2) |
| NE | 21 | (11, 11) | (10, 10) |

In MiniGrid, the agent navigates to the corner position (CORNER_POSES), then uses the ENTER_WELL action (action 3) to visit the adjacent well position and receive reward.

### Cue-goal orientation

Each subject has a fixed cue-goal orientation (e.g., "N/NE") that determines which cue indicates which goal corner. This stays constant across all sessions for a given subject.

Within an acquisition session, the goal corner is fixed. Across sessions, the goal corner is determined by the trial configuration (which cue is presented and the subject's orientation mapping).

### Trial reward

- Correct well entered during trial: R = +1 (WELL_REWARD_SCR)
- Wrong well entered during trial: R = small negative (step cost)
- Exposure well: R = +1 (but this is exposure reward, NOT trial reward)

## Key constants

### Real maze

| Constant | Value |
|----------|-------|
| pass_time_in_error_zone | 0.010 s |
| pass_time_out_of_error_zone | 2.0 s |
| pass_time_reward_zone | 0.250 s |
| Coordinate grid | 240 × 240 px |
| Tracking rate | 15 or 30 Hz |

### MiniGrid

| Constant | Value | Notes |
|----------|-------|-------|
| PRETRIAL_MIN_STEPS | 7 | Minimum steps in pretrial before trigger activates |
| STEP_FORWARD_COST | -0.0005 | Default (overridden to 0.0 for SR) |
| STEP_TURN_COST | -0.001 | Default (overridden to 0.0 for SR) |
| WELL_REWARD_SCR | 1.061 | Default (overridden to 1.0 for SR) |
| MAX_STEPS | 6000 | Typical episode step limit |
| Grid size | 13×13 | With walls on perimeter |
| Valid floor positions | 49 | Reachable maze positions |
| Directions | 4 | 0=right, 1=down, 2=left, 3=up |

## Implications for SR modeling

1. **Exposure vs trial reward:** w should ONLY learn from trial-phase rewards. Exposure rewards are non-discriminative (all 4 wells rewarded).
2. **Goal rotation:** The correct well depends on the trial's cue. A position-only SR without cue information can only learn for a single fixed goal. To handle cue-dependent goals, cue must be part of the state representation.
3. **Session structure:** The full pretrial→trial→ITI loop is ~21-33 steps (optimal path). SR with teleports simplifies to just the trial decision; without teleports, the agent must navigate all phases.
4. **Barrier changes:** Barriers differ between pretrial (dead-end), trial (choice corridors), and ITI (open). A single SR M matrix across all phases faces non-stationary transitions (Momennejad 2017, Russek 2017).
