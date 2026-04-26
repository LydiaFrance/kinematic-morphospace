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

echo "  → Registering Jupyter kernel for VS Code auto-selection..."
uv run python -m ipykernel install --user \
    --name python3 \
    --display-name "kinematic-morphospace" >/dev/null
echo "  ✓ kernel registered as 'kinematic-morphospace' (id=python3)"
echo

echo "  → Installing Microsoft core fonts (incl. Andale Mono)..."
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" \
    | sudo debconf-set-selections
sudo apt-get update -qq
sudo apt-get install -y -qq ttf-mscorefonts-installer >/dev/null 2>&1 \
    && fc-cache -f >/dev/null 2>&1 \
    && rm -rf "$HOME/.cache/matplotlib" \
    && echo "  ✓ fonts installed (matplotlib cache cleared)" \
    || echo "  ⚠ font install failed — notebooks will fall back to default monospace"
echo

# Install the welcome banner so it prints AFTER Codespaces' default
# welcome message and AFTER the venv auto-activation, ending up at the
# bottom of the terminal next to the prompt.
MARKER="# kinematic-morphospace welcome banner"
if ! grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<'EOF'

# kinematic-morphospace welcome banner
if [ -z "${KMS_BANNER_SHOWN:-}" ] && [ -f "/workspaces/kinematic-morphospace/.devcontainer/post-attach.sh" ]; then
    export KMS_BANNER_SHOWN=1
    bash /workspaces/kinematic-morphospace/.devcontainer/post-attach.sh
fi
EOF
fi
