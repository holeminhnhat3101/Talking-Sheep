#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

cd "$ROOT_DIR"

if [[ "$(getconf LONG_BIT)" != "64" ]]; then
    echo "Error: Talking Sheep requires 64-bit Raspberry Pi OS."
    exit 1
fi

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Error: Python 3 was not found."
    echo "Install it with:"
    echo "  sudo apt install python3 python3-venv"
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment at $VENV_DIR..."

    if ! "$PYTHON_COMMAND" -m venv "$VENV_DIR"; then
        echo "Error: Unable to create the virtual environment."
        echo "Install venv support with:"
        echo "  sudo apt install python3-venv"
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

export PHOGPT_MODEL_PATH="${PHOGPT_MODEL_PATH:-$ROOT_DIR/models/PhoGPT-4B-Chat-Q4_K_M.gguf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

dependencies_ready() {
    "$PYTHON" - <<'PY'
import importlib.util
import sys

required = [
    "llama_cpp",
    "kokoro_vietnamese",
    "numpy",
    "pydub",
    "pyaudio",
    "sounddevice",
    "whisper",
]

missing = [
    package
    for package in required
    if importlib.util.find_spec(package) is None
]

if missing:
    print("Missing Python packages:", ", ".join(missing))
    sys.exit(1)
PY
}

if ! dependencies_ready; then
    echo "Installing Raspberry Pi dependencies..."

    "$PYTHON" -m pip install --upgrade pip setuptools wheel
    "$PIP" install \
        --prefer-binary \
        --no-input \
        -r "$ROOT_DIR/requirements-rpi.txt"

    "$PIP" install \
        --prefer-binary \
        --no-input \
        -e "$ROOT_DIR/Kokoro-Vietnamese[onnx]"
fi

echo "Checking PhoGPT Q4 model..."

"$PYTHON" - <<'PY'
from src.chat_phogpt import ensure_model

model_path = ensure_model()
print(f"PhoGPT model ready: {model_path}")
PY

echo "Starting Talking Sheep..."

exec "$PYTHON" "$ROOT_DIR/src/talking_sheep_voice.py" "$@"