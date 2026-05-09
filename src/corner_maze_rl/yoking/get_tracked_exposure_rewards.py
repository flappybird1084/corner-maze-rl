"""Reconstruct exposure reward events from parquet tracking data.

For each reward event in an exposure session, identifies which well the rat
was visiting by checking sess_zone (if it's a well zone) or looking backward
through cord_zone tracking data. Tracks cycle-alternation reward availability:
all 4 wells start available, each visited well is removed from the set, and
the set resets once all 4 have been visited (gen_exposure logic).

Zone-to-well mapping:
    Zone 1  -> SW (1, 11)
    Zone 5  -> NW (1, 1)
    Zone 17 -> SE (11, 11)
    Zone 21 -> NE (11, 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Zone -> one-hot index in [NE, SE, SW, NW]
ZONE_TO_WELL_IDX = {21: 0, 17: 1, 1: 2, 5: 3}
WELL_ZONES = frozenset(ZONE_TO_WELL_IDX.keys())
WELL_NAMES = ['NE', 'SE', 'SW', 'NW']


def load_exposure_session(
    parquet_path: str,
    subject: str = 'CM005',
    session_type: str = 'Exposure',
    session_number: str = '1e',
    df: pd.DataFrame | None = None,
) -> pd.Series:
    """Load a single exposure session row from the parquet file.

    Args:
        parquet_path: Path to the parquet file (ignored if df is provided).
        subject: Subject identifier.
        session_type: Session type filter.
        session_number: Session number filter ('1e' or '2e').
        df: Pre-loaded parquet DataFrame. If None, reads from parquet_path.
    """
    if df is None:
        df = pd.read_parquet(parquet_path)
    mask = (
        (df['subject'] == subject)
        & (df['session_type'] == session_type)
        & (df['session_number'] == session_number)
    )
    matches = df[mask]
    if len(matches) == 0:
        raise ValueError(
            f"No rows found for subject={subject}, "
            f"session_type={session_type}, session_number={session_number}"
        )
    return matches.iloc[0]


def find_all_candidates(
    sess_zone_val: int,
    cord_zone: np.ndarray,
    cord_ts: np.ndarray,
    event_ts: float,
    sess_dur: float,
    prev_center_ts: float | None = None,
    next_center_ts: float | None = None,
) -> list[tuple[int, float]]:
    """Find all well zone visits within the bounded search window.

    Returns ALL candidate wells (not just available ones), sorted by
    distance from the estimated reward time. Each well appears at most
    once (the closest entry for that well).

    Args:
        sess_zone_val: Zone value from sess_zone column for this event.
        cord_zone: Full array of zone integers from tracking data.
        cord_ts: Full array of timestamps (ms) from tracking data.
        event_ts: sess_time_stamp value (ms), the recording time.
        sess_dur: sess_time_dur value (seconds) for this event.
        prev_center_ts: Estimated reward time of previous event (for boundary).
        next_center_ts: Estimated reward time of next event (for boundary).

    Returns:
        List of (well_idx, cord_timestamp) sorted by distance from center.
        Each well_idx appears at most once (closest entry wins).
    """
    center_ts = event_ts - (sess_dur + 45) * 1000

    # Compute search boundaries
    min_ts = -np.inf
    max_ts = np.inf
    if prev_center_ts is not None:
        min_ts = prev_center_ts
    if next_center_ts is not None:
        max_ts = next_center_ts

    # If sess_zone is a well zone, include it as a candidate at center_ts
    best_per_well: dict[int, tuple[float, float]] = {}  # well_idx -> (distance, cord_ts)
    if sess_zone_val in WELL_ZONES:
        idx = ZONE_TO_WELL_IDX[sess_zone_val]
        best_per_well[idx] = (0.0, center_ts)

    # Ripple search through cord_zone
    center_idx = int(np.searchsorted(cord_ts, center_ts))
    n = len(cord_zone)

    for step in range(n):
        for k in [center_idx - step, center_idx + step]:
            if 0 <= k < n:
                t = cord_ts[k]
                if t < min_ts or t > max_ts:
                    continue
                z = int(cord_zone[k])
                if z in WELL_ZONES:
                    idx = ZONE_TO_WELL_IDX[z]
                    dist = abs(t - center_ts)
                    if idx not in best_per_well or dist < best_per_well[idx][0]:
                        best_per_well[idx] = (dist, t)
        # Stop if both sides are past boundaries
        if center_idx - step < 0 or (step > 0 and cord_ts[max(center_idx - step, 0)] < min_ts):
            below_done = True
        else:
            below_done = False
        if center_idx + step >= n or cord_ts[min(center_idx + step, n - 1)] > max_ts:
            above_done = True
        else:
            above_done = False
        if below_done and above_done:
            break

    # Sort by distance from center
    result = [(idx, ts) for idx, (dist, ts) in sorted(
        best_per_well.items(), key=lambda x: x[1][0]
    )]
    return result


def find_well_in_window(
    cord_zone: np.ndarray,
    cord_ts: np.ndarray,
    lo: float,
    hi: float,
    wells_remaining: set[int],
) -> tuple[int, float]:
    """Find the rewarded well in a per-reward window [lo, hi].

    Searches cord_zone within the window for well zone entries.
    Prefers a well that is still available in the current cycle.
    If no available well is found, returns the earliest well entry
    (the cycle will be reset by the caller).

    Args:
        cord_zone: Full array of zone integers from tracking data.
        cord_ts: Full array of timestamps (ms) from tracking data.
        lo: Window start (prev sess_ts or session start).
        hi: Window end (current sess_ts).
        wells_remaining: Wells still available in current cycle.

    Returns:
        (well_idx, well_entry_ts).
    """
    start = int(np.searchsorted(cord_ts, lo))
    end = int(np.searchsorted(cord_ts, hi, side='right'))

    # Collect earliest entry per well in the window
    earliest: dict[int, float] = {}
    for k in range(start, min(end, len(cord_zone))):
        z = int(cord_zone[k])
        if z in WELL_ZONES:
            idx = ZONE_TO_WELL_IDX[z]
            if idx not in earliest:
                earliest[idx] = cord_ts[k]

    # Prefer an available well (earliest among available)
    best_avail = None
    for widx, ts in sorted(earliest.items(), key=lambda x: x[1]):
        if widx in wells_remaining:
            best_avail = (widx, ts)
            break

    if best_avail is not None:
        return best_avail

    # No available well — return earliest well found (triggers cycle issue)
    if earliest:
        first = min(earliest.items(), key=lambda x: x[1])
        return first

    # Should not happen if every window has a well, but fallback
    return None


def identify_well(
    sess_zone_val: int,
    cord_zone: np.ndarray,
    cord_ts: np.ndarray,
    event_ts: float,
    wells_remaining: set[int],
    sess_dur: float = 0.0,
    prev_center_ts: float | None = None,
    next_center_ts: float | None = None,
) -> tuple[int, float] | None:
    """Identify which available well the rat visited for a reward event.

    Only returns wells that are still in wells_remaining (i.e. have not
    been consumed in the current cycle). Strategy:
      1. Check sess_zone directly (uses estimated reward time as timestamp)
      2. Ripple search outward from the estimated reward time
         (event_ts - (sess_dur + 45) * 1000), bounded to stay within
         this event's territory (not past neighboring events' centers)
      3. If only one well remains, assign it (uses estimated reward time)

    Args:
        sess_zone_val: Zone value from sess_zone column for this event.
        cord_zone: Full array of zone integers from tracking data.
        cord_ts: Full array of timestamps (ms) from tracking data.
        event_ts: sess_time_stamp value (ms), the recording time.
        wells_remaining: Set of well indices still available in current cycle.
        sess_dur: sess_time_dur value (seconds) for this event.
        prev_center_ts: Estimated reward time of previous event (for boundary).
        next_center_ts: Estimated reward time of next event (for boundary).

    Returns:
        (well_idx, well_entry_ts) or None if not found.
        well_idx: 0=NE, 1=SE, 2=SW, 3=NW.
        well_entry_ts: cord_zone timestamp (ms) of the well entry.
    """
    center_ts = event_ts - (sess_dur + 45) * 1000

    # 1. If sess_zone is an available well zone, use it directly
    if sess_zone_val in WELL_ZONES:
        idx = ZONE_TO_WELL_IDX[sess_zone_val]
        if idx in wells_remaining:
            return idx, center_ts

    # 2. Ripple search from estimated reward time
    center_idx = int(np.searchsorted(cord_ts, center_ts))
    n = len(cord_zone)

    # Compute search boundaries: up to but not past neighboring centers
    min_ts = -np.inf
    max_ts = np.inf
    if prev_center_ts is not None:
        min_ts = prev_center_ts
    if next_center_ts is not None:
        max_ts = next_center_ts

    for step in range(n):
        for k in [center_idx - step, center_idx + step]:
            if 0 <= k < n:
                t = cord_ts[k]
                if t < min_ts or t > max_ts:
                    continue
                z = int(cord_zone[k])
                if z in WELL_ZONES:
                    idx = ZONE_TO_WELL_IDX[z]
                    if idx in wells_remaining:
                        return idx, t
        # Stop if we've searched beyond both boundaries and array ends
        if center_idx - step < 0 or (step > 0 and cord_ts[max(center_idx - step, 0)] < min_ts):
            below_done = True
        else:
            below_done = False
        if center_idx + step >= n or cord_ts[min(center_idx + step, n - 1)] > max_ts:
            above_done = True
        else:
            above_done = False
        if below_done and above_done:
            break

    # 3. If only one well remains, it must be this one (reward confirms visit)
    if len(wells_remaining) == 1:
        return next(iter(wells_remaining)), center_ts

    return None


def get_tracked_exposure_rewards(
    parquet_path: str,
    subject: str = 'CM005',
    session_type: str = 'Exposure',
    session_number: str = '1e',
    df: pd.DataFrame | None = None,
) -> list[tuple[float, int, list[int], int | None, int | None, float | None]]:
    """Reconstruct exposure reward sequence with cycle-alternation tracking.

    Supports both 1e (32 rewards, all events are rewards) and 2e (33 rewards,
    first 6 events are barrier-drop acclimation, rewards start at index 6).

    Returns:
        List of (timestamp_ms, reward_count, [NE, SE, SW, NW] availability,
        well_idx, well_zone, dwell_time_ms).
        The availability vector uses 1 for wells that still have reward
        available, 0 for already-visited wells in the current cycle. Resets
        when all 4 wells have been visited. well_idx is 0=NE, 1=SE, 2=SW, 3=NW.
        well_zone is the actual zone number (1, 5, 17, or 21).
        dwell_time_ms is how long (ms) the rat stayed in the well zone.
    """
    row = load_exposure_session(parquet_path, subject, session_type, session_number, df=df)

    cord_zone = np.array(row['cord_zone'], dtype=int)
    cord_ts = np.array(row['cord_time_stamp'], dtype=float)
    sess_ts = np.array(row['sess_time_stamp'], dtype=float)

    # Fix outlier cord_ts[0] (some sessions have a bogus first timestamp)
    if len(cord_ts) > 1 and cord_ts[0] > cord_ts[1] * 10:
        cord_ts[0] = cord_ts[1]

    # For 2e, first 6 sess events are barrier-drop acclimation, not rewards
    reward_start_idx = 6 if session_number == '2e' else 0

    results: list[tuple[float, int, list[int], int | None, int | None, float | None]] = []

    reward_indices = list(range(reward_start_idx, len(sess_ts)))
    num_rewards = len(reward_indices)

    # Sequential scan: mimic the reward system by scanning cord_zone forward.
    # For each reward, find the next available well zone entry. The reward
    # fires when the rat first enters an available well after the previous reward.
    wells_remaining = {0, 1, 2, 3}
    reward_count = 0
    cursor = 0  # current position in cord_zone
    prev_reward_ts = -np.inf  # enforce minimum gap between rewards

    for k in range(num_rewards):
        # Scan forward from cursor to find first available well entry
        # that is at least 10s after the previous reward
        well_idx = None
        well_entry_ts = None
        well_zone = None
        dwell_time = None
        min_ts = prev_reward_ts + 12_000  # 12s minimum inter-reward gap
        for j in range(cursor, len(cord_zone)):
            if cord_ts[j] < min_ts:
                continue
            z = int(cord_zone[j])
            if z in WELL_ZONES:
                idx = ZONE_TO_WELL_IDX[z]
                if idx in wells_remaining:
                    well_idx = idx
                    well_entry_ts = cord_ts[j]
                    well_zone = z
                    # Compute dwell time: how long the rat stays in this zone
                    stay = j + 1
                    while stay < len(cord_zone) and cord_zone[stay] == z:
                        stay += 1
                    exit_ts = cord_ts[min(stay, len(cord_zone) - 1)]
                    dwell_time = exit_ts - well_entry_ts
                    cursor = j + 1
                    break

        reward_count += 1
        if well_idx is not None:
            wells_remaining.discard(well_idx)
            prev_reward_ts = well_entry_ts

        avail = [1 if w in wells_remaining else 0 for w in range(4)]
        results.append((well_entry_ts, reward_count, avail, well_idx, well_zone, dwell_time))

        # Reset cycle when all 4 wells consumed
        if not wells_remaining:
            wells_remaining = {0, 1, 2, 3}

    return results


MAX_REWARDS = 33  # 2e has 33 rewards, 1e has 32


def build_exposure_reward_dataframe(
    parquet_path: str,
) -> pd.DataFrame:
    """Build a dataframe of exposure reward times for all valid subjects.

    Only includes subjects with exactly one 1e and one 2e exposure session.
    Each row is one subject + session. Columns reward_1..reward_33 contain
    tuples of (timestamp_ms, reward_number, [NE, SE, SW, NW] availability)
    or None if that reward slot doesn't exist (e.g. reward_33 for 1e sessions).

    Returns:
        DataFrame with columns: subject, session_number, reward_1..reward_33.
    """
    df = pd.read_parquet(parquet_path)

    # Find subjects with exactly one 1e and one 2e
    exp = df[df['session_type'] == 'Exposure']
    counts = exp.groupby(['subject', 'session_number']).size().unstack(fill_value=0)
    valid_subjects = counts[
        (counts.get('1e', 0) == 1) & (counts.get('2e', 0) == 1)
    ].index.tolist()

    rows = []
    for subject in valid_subjects:
        for sess_num in ('1e', '2e'):
            results = get_tracked_exposure_rewards(
                parquet_path, subject=subject, session_number=sess_num, df=df,
            )

            row_data = {'subject': subject, 'session_number': sess_num}
            event_minutes = []
            for i in range(MAX_REWARDS):
                col = f'reward_{i + 1}'
                if i < len(results):
                    ts, count, avail, well_idx, well_zone, dwell = results[i]
                    row_data[col] = str((ts, int(count), avail, well_zone, dwell))
                    event_minutes.append(round(ts / 60_000, 1))
                else:
                    row_data[col] = None
            row_data['reward_event_list'] = str(event_minutes)

            rows.append(row_data)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys

    parquet_path = sys.argv[1] if len(sys.argv) > 1 else 'data/dataframes/all_sessions.parquet'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'dataframe'

    if mode == 'dataframe':
        reward_df = build_exposure_reward_dataframe(parquet_path)
        out_path = 'data/dataframes/exposure_reward_times.parquet'
        reward_df.to_parquet(out_path, index=False)
        print(f'Saved to {out_path}')
        print(f'Shape: {reward_df.shape}')
        print(f'Subjects: {reward_df["subject"].nunique()}')
    else:
        # Single session mode: pass session_number as second arg
        session_number = mode
        results = get_tracked_exposure_rewards(
            parquet_path, session_number=session_number,
        )

        print(f"{'Rwd':>3}  {'Timestamp':>12}  {'Well':>4}  Availability [NE, SE, SW, NW]  Cycle")
        print("-" * 65)

        cycle = 0
        prev_avail = [1, 1, 1, 1]
        for ts, count, avail, well_idx, well_zone, dwell in results:
            name = WELL_NAMES[well_idx] if well_idx is not None else '??'
            if prev_avail == [0, 0, 0, 0] and sum(avail) > 0:
                cycle += 1
            ts_str = f"{ts:>12.0f}" if ts is not None else f"{'N/A':>12}"
            print(f"  {count:>2}  {ts_str}  {name:>4}  {avail}  {cycle}")
            prev_avail = avail
