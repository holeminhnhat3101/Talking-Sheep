#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() { printf '[%s] %s\n' "$1" "$2"; }
fail() { log ERROR "$1"; exit 1; }

[[ "$(getconf LONG_BIT)" == "64" ]] \
    || fail "A 64-bit operating system is required."

command -v python3 >/dev/null 2>&1 \
    || fail "python3 is required."

command -v pkg-config >/dev/null 2>&1 \
    || fail "pkg-config is required; install pkg-config first."

SYSTEM_PYTHON="$(command -v python3)"
PY_VERSION="$(
    "$SYSTEM_PYTHON" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
PYTHON_INCLUDE="$(
    "$SYSTEM_PYTHON" -c \
    'import os,sysconfig; print(os.path.join(sysconfig.get_path("include"), "Python.h"))'
)"

native_packages=(
    python3-venv
    python3-dev
    build-essential
    cmake
    pkg-config
    portaudio19-dev
    libasound2-dev
    ffmpeg
    libsndfile1
    alsa-utils
)

missing_native=()
for package in "${native_packages[@]}"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
        | grep -q 'install ok installed' \
        || missing_native+=("$package")
done

if ((${#missing_native[@]})); then
    log INSTALL "Installing native packages: ${missing_native[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_native[@]}" \
        || fail "Native dependency installation failed."
else
    log SKIP "Native packages already installed."
fi

[[ -f "$PYTHON_INCLUDE" ]] \
    || fail "Python.h is missing for Python $PY_VERSION: $PYTHON_INCLUDE"

pkg-config --exists portaudio-2.0 \
    || fail "PortAudio development headers are unavailable."

VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log INSTALL "Creating .venv with $SYSTEM_PYTHON"
    "$SYSTEM_PYTHON" -m venv "$VENV_DIR" \
        || fail "Virtual environment creation failed."
else
    log SKIP "Using existing .venv."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON="$VENV_DIR/bin/python"
VENV_INCLUDE="$(
    "$PYTHON" -c \
    'import os,sysconfig; print(os.path.join(sysconfig.get_path("include"), "Python.h"))'
)"

[[ -f "$VENV_INCLUDE" ]] \
    || fail "Python.h is missing for the venv interpreter: $VENV_INCLUDE"

mkdir -p runtime models assets/bleats

export LLM_MODEL_PATH="${LLM_MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B-Q4_K_M.gguf}"

export STT_MODEL_DIR="${STT_MODEL_DIR:-$ROOT_DIR/models/zipformer-vi-streaming}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

required_modules=(
    llama_cpp
    numpy
    pydub
    pyaudio
    soundfile
    huggingface_hub
    sherpa_onnx
)

missing_modules=()
for module in "${required_modules[@]}"; do
    if ! "$PYTHON" -c \
        "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$module') else 1)"
    then
        missing_modules+=("$module")
    fi
done

if ((${#missing_modules[@]})); then
    log INSTALL "Installing Python dependencies for: ${missing_modules[*]}"

    "$PYTHON" -m pip install --upgrade pip setuptools wheel \
        || fail "Failed to upgrade pip tooling."

    "$PYTHON" -m pip install \
        --prefer-binary \
        --no-input \
        -r "$ROOT_DIR/requirements-rpi.txt" \
        || fail "Python dependency installation failed."
else
    log SKIP "General Python dependencies already installed."
fi

if ! "$PYTHON" -c \
    'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("kokoro_vietnamese") else 1)'
then
    log INSTALL "Installing Kokoro ONNX support."

    "$PYTHON" -m pip install \
        --prefer-binary \
        --no-input \
        -e "$ROOT_DIR/Kokoro-Vietnamese[onnx]" \
        || fail "Kokoro installation failed."
else
    log SKIP "Kokoro ONNX support already installed."
fi

"$PYTHON" - <<'PY' || fail "Zipformer model setup failed."
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = "hynt/Zipformer-30M-RNNT-Streaming-6000h"
model_dir = Path(os.environ["STT_MODEL_DIR"]).resolve()
model_dir.mkdir(parents=True, exist_ok=True)

filenames = (
    "encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    "decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    "joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    "config.json",
    "bpe.model",
)

for filename in filenames:
    destination = model_dir / filename

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[SKIP] {filename}")
        continue

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
    )

    downloaded_path = Path(downloaded)
    if not downloaded_path.is_file() or downloaded_path.stat().st_size == 0:
        raise RuntimeError(f"Invalid downloaded model file: {filename}")

    print(f"[READY] {filename}")
PY

"$PYTHON" - <<'PY' || fail "LLM model setup failed."
from src.chat_llm import ensure_model

ensure_model()
PY

"$PYTHON" - <<'PY' || fail "Model validation failed."
from pathlib import Path

from src.config import (
    STT_DECODER_PATH,
    STT_ENCODER_PATH,
    STT_JOINER_PATH,
    STT_TOKENS_PATH,
)

required = (
    STT_ENCODER_PATH,
    STT_DECODER_PATH,
    STT_JOINER_PATH,
    STT_TOKENS_PATH,
)

missing = [
    str(path)
    for path in required
    if not Path(path).is_file() or Path(path).stat().st_size == 0
]

if missing:
    raise RuntimeError(
        "Missing or empty Zipformer model files: " + ", ".join(missing)
    )
PY

log READY "Dependencies, Zipformer STT, and LLM model are ready."

exec "$PYTHON" -m src.talking_sheep_voice "$@"