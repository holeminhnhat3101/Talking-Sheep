#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

# The application uses relative defaults such as assets/bleats and runtime.
# Always run it from the repository root, even when invoked from elsewhere.
cd "$ROOT_DIR"

if [[ "$(getconf LONG_BIT)" != "64" ]]; then
    echo "This model requires 64-bit Raspberry Pi OS. Install the 64-bit edition and try again."
    exit 1
fi

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3, then run this file again."
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating a Python environment..."
    if ! "$PYTHON_COMMAND" -m venv "$VENV_DIR"; then
        echo "Unable to create a virtual environment. Install Python's venv support and try again."
        echo "On Debian/Raspberry Pi OS, install python3-venv manually, then rerun this script."
        exit 1
    fi
fi
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"

export PHOGPT_MODEL_PATH="${PHOGPT_MODEL_PATH:-$ROOT_DIR/models/PhoGPT-4B-Chat-Q4_K_M.gguf}"

if ! "$PYTHON" -c 'import llama_cpp, kokoro_vietnamese, numpy, pydub, pyaudio, sounddevice, whisper' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --upgrade pip setuptools wheel
    "$PYTHON" -m pip install --prefer-binary --no-input -r "$ROOT_DIR/requirements-rpi.txt"
    "$PYTHON" -m pip install -e "$ROOT_DIR/Kokoro-Vietnamese[onnx]"
fi

# ensure_model validates the configured Q4 path and downloads the GGUF once
# when it is absent. Set PHOGPT_AUTO_DOWNLOAD=0 to disable this behavior.
"$PYTHON" -c 'from src.chat_phogpt import ensure_model; print(f"PhoGPT model ready: {ensure_model()}")'

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" "$ROOT_DIR/src/talking_sheep_voice.py" "$@"