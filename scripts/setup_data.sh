#!/usr/bin/env bash
# Copy read-only data artifacts from the legacy repo into this one.
# Per plan §16.2 (yoked dataset) and §16.3 (encoder dictionaries).
#
# Usage:
#   scripts/setup_data.sh [LEGACY_PATH]
#
# Default LEGACY_PATH is ../corner-maze-rl-legacy relative to this repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGACY="${1:-${REPO_ROOT}/../corner-maze-rl-legacy}"

if [[ ! -d "$LEGACY" ]]; then
    echo "ERROR: legacy repo not found at $LEGACY" >&2
    echo "Pass its path as the first argument." >&2
    exit 1
fi

echo "Legacy repo: $LEGACY"

# Encoder dictionaries (Phase 1 / 2 needs)
mkdir -p "$REPO_ROOT/data/encoders"
cp -v "$LEGACY/2S2C_task/embeddings/60d/position/pose_60Dvector_dictionary.pkl" \
      "$REPO_ROOT/data/encoders/"
if [[ -f "$LEGACY/2S2C_task/embeddings/60d/image/ryans_visual_embedding_dictionary.pkl" ]]; then
    cp -v "$LEGACY/2S2C_task/embeddings/60d/image/ryans_visual_embedding_dictionary.pkl" \
          "$REPO_ROOT/data/encoders/"
fi

# Yoked dataset (3-table normalized form)
mkdir -p "$REPO_ROOT/data/yoked/dataset"
for f in subjects.parquet sessions.parquet \
         actions_synthetic_pretrial.parquet actions_real_pretrial.parquet; do
    src="$LEGACY/data/yoked/dataset/$f"
    if [[ -f "$src" ]]; then
        cp -v "$src" "$REPO_ROOT/data/yoked/dataset/"
    else
        echo "  (skip — not present in legacy: $f)"
    fi
done

echo "Done."
