#!/usr/bin/env bash
set -euo pipefail

# Resolve config path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config/dargus_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config not found at $CONFIG_FILE" >&2
    exit 1
fi

# Parse conda_env and python version from YAML (no external YAML parser needed)
CONDA_ENV=$(grep 'conda_env:' "$CONFIG_FILE" | head -1 | sed 's/.*conda_env: *"\(.*\)"/\1/')
PYTHON_VER=$(grep 'python:' "$CONFIG_FILE" | head -1 | sed 's/.*python: *"\(.*\)"/\1/')

if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: could not parse conda_env from $CONFIG_FILE" >&2
    exit 1
fi
if [ -z "$PYTHON_VER" ]; then
    echo "ERROR: could not parse python version from $CONFIG_FILE" >&2
    exit 1
fi

SKIP_CREATE=0
RECREATE=0

usage() {
    echo "Usage: $0 [--skip-create] [--recreate]"
    echo "  --skip-create   Skip conda env creation, only reinstall dargus"
    echo "  --recreate      Remove and recreate conda env from scratch"
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-create) SKIP_CREATE=1 ;;
        --recreate) RECREATE=1 ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

if [ "$RECREATE" -eq 1 ] && [ "$SKIP_CREATE" -eq 1 ]; then
    echo "ERROR: --recreate and --skip-create are mutually exclusive" >&2
    exit 1
fi

if [ "$RECREATE" -eq 1 ]; then
    echo "Removing existing conda env '$CONDA_ENV'..."
    conda env remove -n "$CONDA_ENV" -y 2>/dev/null || true
fi

if [ "$SKIP_CREATE" -eq 0 ]; then
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "Conda env '$CONDA_ENV' already exists, skipping creation."
        echo "Use --recreate to rebuild from scratch."
    else
        echo "Creating conda env '$CONDA_ENV' (python=$PYTHON_VER)..."
        conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
    fi
fi

# Resolve the workspace root (parent of script's grandparent = dargus package -> workspace)
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Installing dargus into conda env '$CONDA_ENV'..."
conda run -n "$CONDA_ENV" pip install -e "$WORKSPACE_ROOT[all]"

echo ""
echo "Verifying installation..."
conda run -n "$CONDA_ENV" dargus-cli --help > /dev/null 2>&1

# Install shell wrapper as 'dargus' in the conda env's bin
WRAPPER_SRC="${SCRIPT_DIR}/dargus_wrapper"
WRAPPER_DST="$(conda run -n "${CONDA_ENV}" python -c 'import sys; print(sys.prefix)')/bin/dargus"
echo "Installing dargus shell wrapper..."
cp "${WRAPPER_SRC}" "${WRAPPER_DST}"
chmod +x "${WRAPPER_DST}"

echo ""
echo "Setup complete. Run 'dargus' from any shell — it auto-activates the conda environment."
echo "  conda activate $CONDA_ENV   # optional: for faster execution (skips conda overhead)"
