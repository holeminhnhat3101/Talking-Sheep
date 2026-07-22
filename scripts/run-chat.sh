#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    VENV_DIR="$VIRTUAL_ENV"
else
    VENV_DIR="$SCRIPT_DIR/.venv"
fi

PYTHON=""
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

if [[ "$(getconf LONG_BIT)" != "64" ]]; then
    echo "This model requires 64-bit Raspberry Pi OS. Install the 64-bit edition and try again."
    exit 1
fi

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3, then run this file again."
    exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/bin/python3" ]]; then
    PYTHON="$VENV_DIR/bin/python3"
fi

venv_python_is_usable() {
    local candidate="$1"
    [[ -x "$candidate" ]] || return 1
    "$candidate" -c 'import sys; print(sys.version_info[0])' >/dev/null 2>&1
}

if [[ -n "$PYTHON" ]] && ! venv_python_is_usable "$PYTHON"; then
    echo "Found an unusable virtual environment at $VENV_DIR (often happens after copying between machines)."
    echo "Recreating it for this device..."
    rm -rf "$VENV_DIR"
    PYTHON=""
fi

if [[ -z "$PYTHON" && -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi

if [[ -z "$PYTHON" ]]; then
    echo "Creating a Python environment..."
    if ! "$PYTHON_COMMAND" -m venv "$VENV_DIR"; then
        echo "Unable to create a virtual environment. Install Python's venv support and try again."
        echo "On Debian/Raspberry Pi OS, install python3-venv manually, then rerun this script."
        exit 1
    fi
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/bin/python3" ]]; then
    PYTHON="$VENV_DIR/bin/python3"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Python was not found in the virtual environment at: $VENV_DIR"
    exit 1
fi

if ! "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    true
fi

if ! "$PYTHON" -c 'import huggingface_hub, llama_cpp' >/dev/null 2>&1; then
    echo "Installing AI runtime packages. This happens only on the first run..."
    "$PYTHON" -m pip install --upgrade pip setuptools wheel
    "$PYTHON" -m pip install --prefer-binary --no-input -r "$SCRIPT_DIR/requirements-rpi.txt"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" "$SCRIPT_DIR/../src/chat_phogpt_q8.py"