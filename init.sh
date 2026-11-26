#!/usr/bin/env bash
# Set up jenair in this folder, then run it.
#
#   ./init.sh                          set up, then check the first example
#   ./init.sh configs/full.json        set up (if needed), then run jenair with these arguments
#   ./init.sh pull zimage
#   ./init.sh ui                       set up (if needed), then open the web page
#
# What it does, only where something is missing:
#   1. finds Python 3.10+, or installs it (Homebrew on macOS, your package manager on Linux)
#   2. creates ./.venv and installs jenair into it (nothing is installed system-wide)
#   3. copies .env.example to .env
# Safe to run again: finished steps are skipped.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
VENV=".venv"
STAMP="$VENV/.jenair-installed"
MIN_MINOR=10

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Prints the first Python that is 3.10 or newer; fails when there is none.
find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c "import sys; raise SystemExit(sys.version_info < (3, $MIN_MINOR))" \
        >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else
    command -v sudo >/dev/null 2>&1 || die "need root to run: $*"
    say "sudo $*"
    sudo "$@"
  fi
}

# Installs Python (with venv and pip) using whatever this system offers.
install_python() {
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || die "Python 3.$MIN_MINOR+ was not found and Homebrew is not installed.
  Install Python from https://www.python.org/downloads/ (or Homebrew from https://brew.sh),
  then run ./init.sh again."
      say "brew install python"
      brew install python
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        as_root apt-get update
        as_root apt-get install -y python3 python3-venv python3-pip
      elif command -v dnf >/dev/null 2>&1; then
        as_root dnf install -y python3 python3-pip
      elif command -v pacman >/dev/null 2>&1; then
        as_root pacman -Sy --noconfirm python python-pip
      elif command -v zypper >/dev/null 2>&1; then
        as_root zypper --non-interactive install python3 python3-pip
      elif command -v apk >/dev/null 2>&1; then
        as_root apk add python3 py3-pip
      else
        die "Python 3.$MIN_MINOR+ was not found and no known package manager is here.
  Install Python 3.$MIN_MINOR or newer, then run ./init.sh again."
      fi
      ;;
    *)
      die "Python 3.$MIN_MINOR+ was not found. On Windows use WSL, or install Python from
  https://www.python.org/downloads/ and run:  python -m venv .venv  &&  .venv\\Scripts\\pip install -e ."
      ;;
  esac
}

make_venv() {
  local python="$1"
  say "Creating $VENV with $("$python" --version 2>&1)"
  if ! "$python" -m venv "$VENV" >/dev/null 2>&1; then
    rm -rf "$VENV"
    # Debian and Ubuntu ship venv/ensurepip in a separate package.
    if [ "$(uname -s)" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
      as_root apt-get update
      as_root apt-get install -y python3-venv python3-pip
      "$python" -m venv "$VENV"
    else
      die "could not create a virtual environment with $python (is the venv module installed?)"
    fi
  fi
}

# 1. Python
if [ ! -x "$VENV/bin/python" ]; then
  PYTHON="$(find_python || true)"
  if [ -z "$PYTHON" ]; then
    say "Python 3.$MIN_MINOR+ not found: installing it"
    install_python
    hash -r
    PYTHON="$(find_python || true)"
    [ -n "$PYTHON" ] || die "Python was installed but 3.$MIN_MINOR+ is still not on PATH.
  Open a new terminal and run ./init.sh again."
  fi
  make_venv "$PYTHON"
fi

# 2. pip and jenair, inside .venv only
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  say "Installing pip into $VENV"
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 ||
    die "pip is missing from $VENV and ensurepip failed. Install python3-pip / python3-venv, delete $VENV, and run ./init.sh again."
fi
if [ ! -x "$VENV/bin/jenair" ] || [ ! -f "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then
  say "Installing jenair into $VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e .
  touch "$STAMP"
fi

# 3. settings
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  say "Created .env from .env.example (edit it for RunPod or storage keys)"
fi

# Run
if [ "$#" -gt 0 ]; then
  exec "$VENV/bin/jenair" "$@"
fi

say "Ready. Checking the first example (nothing is rendered):"
"$VENV/bin/jenair" configs/basics/01-one-image.json --dry-run
cat <<'EOF'

Next:
  source .venv/bin/activate              # then the command is just:  jenair ...
  jenair configs/basics/01-one-image.json           # start ComfyUI first; the first run downloads the model (~20 GB)
  jenair configs/basics/01-one-image.json --runpod  # no GPU? rent one (RUNPOD_API_KEY in .env)
  jenair ui                                         # or write and render jobs in your browser

Without activating, ./init.sh runs it too:  ./init.sh configs/full.json --dry-run
Examples: configs/README.md
EOF
