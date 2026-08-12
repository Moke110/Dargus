#!/usr/bin/env sh
#
# Dargus one-line installer — curl -LsSf <release>/latest/download/install.sh | sh
#
# Bootstraps the system so a researcher can install Dargus with a single
# command (Spec 1, T9):
#   1. detects Linux / macOS (fails elsewhere)
#   2. installs uv if missing (standalone installer — never touches conda)
#   3. provisions a uv-managed Python 3.11+ when none is available
#   4. runs `uv tool install dargus-cli` (isolated — the existing system or
#      conda Python is never modified)
#   5. prints the next-step hint (`dargus setup`)
#
# The Dargus home data is never touched by the installer.
#
# Failures exit non-zero with a readable message.

set -euo pipefail

log()  { printf '\033[1;34m[dargus]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dargus]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[dargus]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. Platform detection ---------------------------------------------------
OS="$(uname -s)"
case "$OS" in
  Linux)  log "Detected Linux." ;;
  Darwin) log "Detected macOS." ;;
  *)      die "Unsupported platform '$OS'. Dargus installs on Linux and macOS only." ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|arm64|aarch64) ;;
  *) die "Unsupported architecture '$ARCH'." ;;
esac

# ---- 2. Install uv if missing ------------------------------------------------
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  # The astral standalone installer puts uv in ~/.local/bin on Linux/macOS.
  log "uv not found — installing the uv standalone binary."
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "Failed to install uv."
  UV_HOME="${UV_HOME:-$HOME/.local/bin}"
  export PATH="$UV_HOME:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not found on PATH."
fi
log "Using uv: $(command -v uv)"

# ---- 3. Provision a uv-managed Python 3.11+ ----------------------------------
# `uv python install` is a no-op when the interpreter is already present and
# installs into uv's own managed directory — the system/conda Python is never
# touched.
if ! uv python find 3.11 >/dev/null 2>&1; then
  log "No Python 3.11+ found — installing a uv-managed Python."
  uv python install 3.11 || die "Failed to provision a uv-managed Python 3.11."
fi
PYTHON="$(uv python find 3.11)"
log "Using Python: $PYTHON"

# ---- 4. Install the dargus-cli tool ------------------------------------------
log "Installing dargus-cli (this may take a moment)."
uv tool install --python "$PYTHON" dargus-cli || {
  warn "Install failed. Retrying without pinning an interpreter (uv picks one)."
  uv tool install dargus-cli || die "Failed to install dargus-cli. See the error above."
}

command -v dargus >/dev/null 2>&1 || die "dargus installed but not found on PATH. Re-open your shell or add uv's bin directory to PATH."

# ---- 5. Next-step hint --------------------------------------------------------
log "Dargus installed successfully."
log "Next: run 'dargus setup' to initialise your config, API key, D-Base, and session archive."
log "      run 'uv tool upgrade dargus-cli' to update later."
