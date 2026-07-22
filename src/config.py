"""Central application configuration for Talking Sheep.

Edit this file to change model, voice, audio, and sheep-effect settings.
Vendored Kokoro package defaults remain in ``Kokoro-Vietnamese``.
"""

# PhoGPT
PHOGPT_MODEL_REPO = "vinai/PhoGPT-4B-Chat-gguf"
MODEL_FILENAME = "PhoGPT-4B-Chat-Q4_K_M.gguf"
SYSTEM_PROMPT = """Bạn là một chú cừu thân thiện, trả lời bằng tiếng Việt tự nhiên, rõ ràng và ngắn gọn.
Nếu câu hỏi thiếu ngữ cảnh, hãy hỏi lại để làm rõ thay vì đoán."""
PROMPT_TEMPLATE = "### Người dùng: {instruction}\n### Trả lời:"

# Audio format
KOKORO_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit

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
