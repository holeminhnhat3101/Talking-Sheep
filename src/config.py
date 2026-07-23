"""Central application configuration for Talking Sheep.

Edit this file to change model, voice, audio, and sheep-effect settings.
Vendored Kokoro package defaults remain in ``Kokoro-Vietnamese``.
"""

import os


def _float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the valid range")
    return value


def _device(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer device index or unset") from exc

# PhoGPT
PHOGPT_MODEL_REPO = "vinai/PhoGPT-4B-Chat-gguf"
MODEL_FILENAME = "PhoGPT-4B-Chat-Q4_K_M.gguf"
SYSTEM_PROMPT = """Bạn là một chú cừu thân thiện, trả lời bằng tiếng Việt tự nhiên, rõ ràng và ngắn gọn.
Nếu câu hỏi thiếu ngữ cảnh, hãy hỏi lại để làm rõ thay vì đoán."""
PROMPT_TEMPLATE = "### Người dùng: {instruction}\n### Trả lời:"
PHOGPT_MAX_TOKENS = 256
PHOGPT_TEMPERATURE = 0.7
PHOGPT_TOP_P = 0.9
PHOGPT_REPEAT_PENALTY = 1.05
PHOGPT_HISTORY_MAXLEN = 4
PHOGPT_N_BATCH_MAX = 512
PHOGPT_CONTEXT = 2048

# Audio format
KOKORO_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
DEFAULT_INPUT_WAV = "input.wav"
DEFAULT_FINAL_WAV = "final.wav"

# Spoken voice
# Supported Kokoro Vietnamese voices:
#   diem_trinh  — Diễm Trinh
#   hung_thinh  — Hưng Thịnh
#   mai_linh    — Mai Linh
#   mai_loan    — Mai Loan
#   manh_dung   — Mạnh Dũng
#   my_yen      — Mỹ Yến
#   ngoc_huyen  — Ngọc Huyền
#   phat_tai    — Phát Tài
#   thanh_dat   — Thành Đạt
#   thuc_trinh  — Thục Trinh
#   tuan_ngoc   — Tuấn Ngọc
#   storyvert   — storyvert
#   duc_an      — Đức An
#   duc_duy     — Đức Duy
#
'''
Để test voice mà không xài llm, chạy câu lệnh sau trong terminal
cd /Users/(tên)/Talking-Sheep

export CACHE="$HOME/.cache/huggingface/hub/models--contextboxai--Kokoro-Vietnamese/snapshots/9f210d622209fcc216fe2ac6159fed2ff381cb8a"

.venv/bin/python - <<'PY'
import importlib
import sys
from pathlib import Path

root = Path.cwd()
cache = Path.home() / ".cache/huggingface/hub/models--contextboxai--Kokoro-Vietnamese/snapshots/9f210d622209fcc216fe2ac6159fed2ff381cb8a"

sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "Kokoro-Vietnamese" / "src"))

from src.voice_layer import KokoroTTS

KokoroVietnamese = importlib.import_module("kokoro_vietnamese").KokoroVietnamese

engine = KokoroVietnamese(
    device="cpu",
    voice="manh_dung", 
    model_path=cache / "kokoro_vi.pth",
    voicepack_path=cache / "voicepacks" / "manh_dung.pt", 
    config_path=cache / "config.json",
)

tts = KokoroTTS(engine)
output = root / "runtime/voice-test/final.wav"

tts.synthesize(
    "Xin chào! Tôi là một chú cừu thông minh! Hãy khen tôi đi!",
    output_path=output,
)

print(f"Đã tạo: {output}")
PY

afplay runtime/voice-test/final.wav'''


SPEAKING_SPEED = 1.0
DEFAULT_VOICE = "mai_linh"
DEFAULT_DEVICE = "cpu"
DEFAULT_STT_MODEL = "tiny"
WHISPER_ALLOWED_MODELS = ("tiny", "base")

# Sheep bleat effects
SILENCE_MS = 100
BLEAT_FADE_IN_MS = 25
BLEAT_FADE_OUT_MS = 70
BLEAT_VOLUME_DB = -3
PAUSE_BEFORE_BLEAT_MS = SILENCE_MS
PAUSE_AFTER_BLEAT_MS = SILENCE_MS

# Application paths and logging
DEFAULT_BLEATS_DIR = "assets/bleats"
DEFAULT_RUNTIME_DIR = "runtime"
DEFAULT_LOG_LEVEL = "INFO"

# Parsed once at import/startup.
AUDIO_INPUT_DEVICE = _device("AUDIO_INPUT_DEVICE")
AUDIO_OUTPUT_DEVICE = _device("AUDIO_OUTPUT_DEVICE")
SILENCE_THRESHOLD = _float("SILENCE_THRESHOLD", 500, 0)
SILENCE_DURATION = _float("SILENCE_DURATION", 1.5, 0)
MIN_SPEECH_DURATION = _float("MIN_SPEECH_DURATION", 0.4, 0)
MAX_RECORDING_DURATION = _float("MAX_RECORDING_DURATION", 15.0, 0)
PRE_ROLL_DURATION = _float("PRE_ROLL_DURATION", 0.25, 0)
BLEAT_PROBABILITY = _float("BLEAT_PROBABILITY", 0.30, 0, 1)

# Microphone negotiation and capture tuning
AUDIO_CAPTURE_RATE = None
AUDIO_CAPTURE_CHANNELS = None
AUDIO_CHANNEL_MODE = "auto"
AUDIO_AUTO_CALIBRATE = True
AUDIO_CALIBRATION_DURATION = _float("AUDIO_CALIBRATION_DURATION", 0.75, 0)
AUDIO_THRESHOLD_MULTIPLIER = _float("AUDIO_THRESHOLD_MULTIPLIER", 3.0, 0)
AUDIO_MINIMUM_AUTO_THRESHOLD = _float("AUDIO_MINIMUM_AUTO_THRESHOLD", 150.0, 0)
AUDIO_SPEECH_START_CHUNKS = 3
AUDIO_MAX_WAIT_FOR_SPEECH = None
AUDIO_SAVE_NATIVE_DEBUG = False

# Conversation loop retry delays
MIC_RETRY_INITIAL_DELAY = _float("MIC_RETRY_INITIAL_DELAY", 2.0, 0)
MIC_RETRY_MAX_DELAY = _float("MIC_RETRY_MAX_DELAY", 10.0, 0)
CYCLE_RETRY_DELAY = _float("CYCLE_RETRY_DELAY", 1.0, 0)
