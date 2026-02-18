set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$REPO_DIR/src" "$REPO_DIR/.cisopt/bin/python" "$@"
