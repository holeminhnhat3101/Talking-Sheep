"""Cấu hình ứng dụng trung tâm cho Talking Sheep."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là một số") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} phải lớn hơn hoặc bằng {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} phải nhỏ hơn hoặc bằng {maximum}")

    return value


def _int(
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là số nguyên") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} phải lớn hơn hoặc bằng {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} phải nhỏ hơn hoặc bằng {maximum}")

    return value


def _device(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} phải là chỉ số thiết bị dạng số nguyên hoặc để trống"
        ) from exc


# CPU Core detection
CPU_COUNT = os.cpu_count() or 4

# Thread Ratios & Optimization
# Priority 1: Qwen (LLM) - Full cores (100%)
LLM_NUM_THREADS = _int("LLM_NUM_THREADS", CPU_COUNT, 1)

# Priority 2: Kokoro (TTS) & OMP Global - ~75% cores
TTS_INTRA_THREADS = _int("TTS_INTRA_THREADS", max(1, int(CPU_COUNT * 0.75)), 1)
TTS_INTER_THREADS = _int("TTS_INTER_THREADS", 1, 1)

OMP_NUM_THREADS_VAL = _int("OMP_NUM_THREADS", max(1, int(CPU_COUNT * 0.75)), 1)
OPENBLAS_NUM_THREADS_VAL = _int("OPENBLAS_NUM_THREADS", 1, 1)

# Priority 3: Zipformer STT - Fixed low core count
STT_NUM_THREADS = _int("STT_NUM_THREADS", 2, 1)


# Local LLM
LLM_MODEL_REPO = "ggml-org/Qwen3-1.7B-GGUF"
LLM_MODEL_FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"

LLM_SYSTEM_PROMPT = """Bạn là một con cừu thân thiện tên là Cừu.
Luôn trả lời bằng tiếng Việt tự nhiên.
LUÔN VIẾT ĐẦY ĐỦ DẤU, TỪ VÀ CÂU, không dùng teencode, viết tắt, emoji hoặc ký hiệu mạng xã hội.
Nếu lời nói của người dùng có lỗi nhận dạng giọng nói nhưng vẫn suy ra được ý định, hãy tự sửa và trả lời theo ý định đó.
Không nhắc đến việc sửa lỗi nhận dạng giọng nói.
Không xin lỗi hoặc yêu cầu người dùng nói lại nếu đã hiểu được ý chính.
Không bịa đặt thông tin; nếu không biết thì nói không biết.
Giữ giọng điệu ấm áp, thân thiện và tự nhiên như đang trò chuyện.
Viết tối đa 3 câu.
/no_think"""

LLM_MAX_TOKENS = 64
LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.95
LLM_REPEAT_PENALTY = 1.0
LLM_HISTORY_MAXLEN = 4
LLM_N_BATCH_MAX = 256
LLM_CONTEXT = 1024


# Native-streaming Vietnamese STT: Zipformer + sherpa-onnx
STT_MODEL_REPO = "hynt/Zipformer-30M-RNNT-Streaming-6000h"

STT_MODEL_DIR = Path(
    os.environ.get(
        "STT_MODEL_DIR",
        REPO_ROOT / "models" / "zipformer-vi-streaming",
    )
).expanduser().resolve()

STT_ENCODER_FILENAME = (
    "encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
)
STT_DECODER_FILENAME = (
    "decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
)
STT_JOINER_FILENAME = (
    "joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
)

# The repository renamed its token table from tokens.txt to config.json.
STT_TOKENS_FILENAME = "config.json"
STT_BPE_FILENAME = "bpe.model"

STT_ENCODER_PATH = STT_MODEL_DIR / STT_ENCODER_FILENAME
STT_DECODER_PATH = STT_MODEL_DIR / STT_DECODER_FILENAME
STT_JOINER_PATH = STT_MODEL_DIR / STT_JOINER_FILENAME
STT_TOKENS_PATH = STT_MODEL_DIR / STT_TOKENS_FILENAME
STT_BPE_PATH = STT_MODEL_DIR / STT_BPE_FILENAME

STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1

STT_PROVIDER = os.environ.get("STT_PROVIDER", "cpu").strip() or "cpu"
STT_DECODING_METHOD = os.environ.get(
    "STT_DECODING_METHOD",
    "greedy_search",
).strip() or "greedy_search"

STT_MAX_ACTIVE_PATHS = _int("STT_MAX_ACTIVE_PATHS", 4, 1)
STT_ENABLE_ENDPOINT_DETECTION = False
STT_LOG_PARTIALS = os.environ.get(
    "STT_LOG_PARTIALS",
    "",
).strip().lower() in {"1", "true", "yes"}


# Audio formats
KOKORO_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

AUDIO_CHUNK_SIZE = 1024
DEFAULT_INPUT_WAV = "input.wav"
DEFAULT_FINAL_WAV = "final.wav"


# Kokoro voice
SPEAKING_SPEED = 1.6
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
