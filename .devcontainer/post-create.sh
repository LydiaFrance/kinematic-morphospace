#!/usr/bin/env bash
# Runs once when the Codespace is built. Non-interactive — no prompts here.
set -euo pipefail

echo
echo "  ╭─────────────────────────────────────────────────────────╮"
echo "  │  kinematic-morphospace — Codespace setup                │"
echo "  ╰─────────────────────────────────────────────────────────╯"
echo

echo "  → Installing dependencies (uv sync)..."
uv sync --frozen
echo "  ✓ environment ready"
echo
