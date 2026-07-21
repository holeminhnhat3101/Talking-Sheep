#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

if [[ "$(getconf LONG_BIT)" != "64" ]]; then
    echo "This model requires 64-bit Raspberry Pi OS. Install the 64-bit edition and try again."
    exit 1
fi

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3, then run this file again."
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Creating a Python environment..."
    if ! "$PYTHON_COMMAND" -m venv "$VENV_DIR"; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "Installing the Raspberry Pi Python virtual-environment support..."
            sudo apt-get update
            sudo apt-get install -y python3-venv python3-pip
            "$PYTHON_COMMAND" -m venv "$VENV_DIR"
        else
            echo "Unable to create a virtual environment. Install Python's venv support and try again."
            exit 1
        fi
    fi
fi

if ! "$PYTHON" -c 'import huggingface_hub, llama_cpp' >/dev/null 2>&1; then
    echo "Installing AI runtime packages. This happens only on the first run..."
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install --prefer-binary -r "$SCRIPT_DIR/requirements-rpi.txt"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" "$SCRIPT_DIR/chat_phogpt_q8.py"