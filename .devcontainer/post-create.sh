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

echo "  → Installing fonts (Andale Mono via Microsoft Core Fonts)..."
# Run in a subshell so any apt failure here does NOT abort post-create
# (notebooks still render with the default monospace fallback if install fails).
(
    set +e
    # ttf-mscorefonts-installer lives in Debian 'contrib', not the default 'main'.
    if [ -f /etc/apt/sources.list ]; then
        sudo sed -i -E 's/^(deb .* main)$/\1 contrib/' /etc/apt/sources.list || true
    fi
    for f in /etc/apt/sources.list.d/*.sources; do
        [ -f "$f" ] || continue
        sudo sed -i -E 's/^(Components: .*\bmain\b)$/\1 contrib/' "$f" || true
    done
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" \
        | sudo debconf-set-selections
    sudo apt-get update -qq >/dev/null 2>&1
    sudo apt-get install -y -qq ttf-mscorefonts-installer fonts-liberation >/dev/null 2>&1
    rc=$?
    fc-cache -f >/dev/null 2>&1 || true
    rm -rf "$HOME/.cache/matplotlib" || true
    if [ $rc -eq 0 ]; then
        echo "  ✓ fonts installed (matplotlib cache cleared)"
    else
        echo "  ⚠ font install failed — notebooks will fall back to default monospace"
    fi
)
echo

echo "  → Installing Chromium (kaleido 1.x needs a browser for plotly write_image)..."
(
    set +e
    sudo apt-get install -y -qq chromium chromium-driver >/dev/null 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ✓ chromium installed"
    else
        echo "  ⚠ chromium install failed — plotly figure exports (write_image) will not work"
    fi
)
echo

echo "  → Warming import + font caches (avoids 30-90s stall on first notebook run)..."
uv run python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kinematic_morphospace  # noqa: F401
# Force matplotlib to build its font cache now, not on first notebook cell.
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
plt.close(fig)
print('  ✓ caches warm')
" || echo "  ⚠ warmup failed (notebooks will rebuild caches on first run)"
echo

# Install the welcome banner so it prints AFTER Codespaces' default
# welcome message and AFTER the venv auto-activation, ending up at the
# bottom of the terminal next to the prompt.
MARKER="# kinematic-morphospace welcome banner"
if ! grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<'EOF'

# kinematic-morphospace welcome banner
# Hooked into PROMPT_COMMAND so it prints AFTER any venv auto-activation
# that Codespaces / VS Code injects after .bashrc has run.
_kms_show_banner() {
    if [ -z "${KMS_BANNER_SHOWN:-}" ] && [ -f "/workspaces/kinematic-morphospace/.devcontainer/post-attach.sh" ]; then
        export KMS_BANNER_SHOWN=1
        bash /workspaces/kinematic-morphospace/.devcontainer/post-attach.sh
    fi
}
PROMPT_COMMAND="_kms_show_banner;${PROMPT_COMMAND:-}"
EOF
fi
