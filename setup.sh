#!/usr/bin/env bash
set -euo pipefail

echo "── Setting up lumis-eval ──"

# 1. Check uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

# 2. Create virtualenv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# 3. Sync all workspace packages
echo "Installing workspace packages..."
uv sync

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
echo "✓ Or use: uv run <command>"
echo ""
echo "Quick start:"
echo "  cp .env.example .env   # fill in API keys"
echo "  lumiseval --help       # CLI entry point"
echo "  make api               # start REST API"
