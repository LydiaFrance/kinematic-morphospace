#!/usr/bin/env bash
# Runs every time you attach a terminal to the Codespace.
# Friendly status banner + next-step hints for reviewers.
set -euo pipefail

DATA_SENTINEL="data/raw/unlabelled_markers.csv"

echo
echo "  ╭─────────────────────────────────────────────────────────╮"
echo "  │  Welcome — kinematic morphospace reviewer environment   │"
echo "  ╰─────────────────────────────────────────────────────────╯"
echo

if [ -f "$DATA_SENTINEL" ]; then
    echo "  ✓ dataset present in ./data/"
    echo
    echo "  Open any notebook under examples/hawks/ to start."
    echo "  Suggested order: 00_ExperimentalSetup → 15_AlternativeMethods."
    echo
    echo "  Or launch full JupyterLab in the browser:"
    echo "      uv run jupyter lab --no-browser --ip 0.0.0.0 --port 8888"
else
    echo "  ⚠  Dataset not yet downloaded."
    echo
    echo "  Run this in the terminal and paste the share token from"
    echo "  the manuscript's Data Availability section when prompted:"
    echo
    echo "      uv run python scripts/download_figshare_data.py"
    echo
    echo "  (~2.3 GB, 5–10 min on Codespaces.)"
fi
echo
