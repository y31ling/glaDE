#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

GLADE_PORT="${GLADE_PORT:-6017}"
export GLADE_PORT

echo "============================================================"
echo "  GLADE WebUI"
echo "  Open in browser: http://localhost:${GLADE_PORT}"
echo "  Press Ctrl+C to stop"
echo "============================================================"
python3 "${SCRIPT_DIR}/web/app.py"
