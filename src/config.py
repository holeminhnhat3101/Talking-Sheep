"""Central application configuration for Talking Sheep."""

import os


def _float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")

    return value


def _device(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer device index or unset"
        ) from exc


# Local LLM
LLM_MODEL_REPO = "ggml-org/Qwen3-1.7B-GGUF"
LLM_MODEL_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"

LLM_SYSTEM_PROMPT = """Bạn là một con cừu thân thiện.
Tự sửa lỗi nhận dạng giọng nói khi ý nghĩa vẫn rõ.
Trả lời trực tiếp bằng tiếng Việt.
Không xin lỗi hoặc yêu cầu nói lại nếu đã hiểu chủ đề.
Trả lời từ 1 đến 5 câu tùy độ phức tạp.
/no_think"""

LLM_MAX_TOKENS = 128
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_REPEAT_PENALTY = 1.05
LLM_HISTORY_MAXLEN = 4
LLM_N_BATCH_MAX = 256
LLM_CONTEXT = 1024


# Speech-to-text
STT_DEFAULT_MODEL = "tiny"
STT_ALLOWED_MODELS = ("tiny", "base")

STT_MODEL_IDS = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
}


# Audio format
KOKORO_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1

AUDIO_CHUNK_SIZE = 1024
DEFAULT_INPUT_WAV = "input.wav"
DEFAULT_FINAL_WAV = "final.wav"


# Kokoro voice
SPEAKING_SPEED = 1.0
DEFAULT_VOICE = "mai_linh"
DEFAULT_DEVICE = "cpu"


# Sheep effects
SILENCE_MS = 100
BLEAT_FADE_IN_MS = 25
BLEAT_FADE_OUT_MS = 70
BLEAT_VOLUME_DB = -3

PAUSE_BEFORE_BLEAT_MS = SILENCE_MS
PAUSE_AFTER_BLEAT_MS = SILENCE_MS
BLEAT_PROBABILITY = _float("BLEAT_PROBABILITY", 1.0, 0, 1)


# Paths and logging
DEFAULT_BLEATS_DIR = "assets/bleats"
DEFAULT_RUNTIME_DIR = "runtime"
DEFAULT_LOG_LEVEL = "INFO"


# Microphone
AUDIO_INPUT_DEVICE = _device("AUDIO_INPUT_DEVICE")
AUDIO_OUTPUT_DEVICE = _device("AUDIO_OUTPUT_DEVICE")

SILENCE_THRESHOLD = _float("SILENCE_THRESHOLD", 250, 0)
SILENCE_DURATION = _float("SILENCE_DURATION", 1.5, 0)
MIN_SPEECH_DURATION = _float("MIN_SPEECH_DURATION", 0.4, 0)
MAX_RECORDING_DURATION = _float("MAX_RECORDING_DURATION", 15.0, 0)
PRE_ROLL_DURATION = _float("PRE_ROLL_DURATION", 0.25, 0)

AUDIO_CAPTURE_RATE = None
AUDIO_CAPTURE_CHANNELS = None
AUDIO_CHANNEL_MODE = "auto"
AUDIO_AUTO_CALIBRATE = True

AUDIO_CALIBRATION_DURATION = _float(
    "AUDIO_CALIBRATION_DURATION",
    0.75,
    0,
)
AUDIO_THRESHOLD_MULTIPLIER = _float(
    "AUDIO_THRESHOLD_MULTIPLIER",
    3.0,
    0,
)
AUDIO_MINIMUM_AUTO_THRESHOLD = _float(
    "AUDIO_MINIMUM_AUTO_THRESHOLD",
    150.0,
    0,
)

AUDIO_SPEECH_START_CHUNKS = 3
AUDIO_MAX_WAIT_FOR_SPEECH = None
AUDIO_SAVE_NATIVE_DEBUG = False


# Retry delays
MIC_RETRY_INITIAL_DELAY = _float(
    "MIC_RETRY_INITIAL_DELAY",
    2.0,
    0,
)
MIC_RETRY_MAX_DELAY = _float(
    "MIC_RETRY_MAX_DELAY",
    10.0,
    0,
)
CYCLE_RETRY_DELAY = _float(
    "CYCLE_RETRY_DELAY",
    1.0,
    0,
)