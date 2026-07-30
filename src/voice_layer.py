"""Lớp giọng nói: chia câu, tổng hợp TTS, chèn tiếng cừu, soạn audio."""

import logging
import re
import wave
import random
from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment

try:
    from .config import (
        BLEAT_FADE_IN_MS,
        BLEAT_FADE_OUT_MS,
        BLEAT_VOLUME_DB,
        BLEAT_PROBABILITY,
        DEFAULT_BLEATS_DIR,
        DEFAULT_FINAL_WAV,
        DEFAULT_RUNTIME_DIR,
        KOKORO_SAMPLE_RATE,
        PAUSE_AFTER_BLEAT_MS,
        PAUSE_BEFORE_BLEAT_MS,
        SILENCE_MS,
        SPEAKING_SPEED,
        TTS_INTRA_THREADS,
        TTS_INTER_THREADS,
        TARGET_CHANNELS,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_WIDTH,
    )
except ImportError:
    from src.config import (
        BLEAT_FADE_IN_MS,
        BLEAT_FADE_OUT_MS,
        BLEAT_VOLUME_DB,
        BLEAT_PROBABILITY,
        DEFAULT_BLEATS_DIR,
        DEFAULT_FINAL_WAV,
        DEFAULT_RUNTIME_DIR,
        KOKORO_SAMPLE_RATE,
        PAUSE_AFTER_BLEAT_MS,
        PAUSE_BEFORE_BLEAT_MS,
        SILENCE_MS,
        SPEAKING_SPEED,
        TTS_INTRA_THREADS,
        TTS_INTER_THREADS,
        TARGET_CHANNELS,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_WIDTH,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chia câu
# ---------------------------------------------------------------------------

class StreamingSentenceAssembler:
    """Stateful sentence assembler that aggregates stream chunks and yields
    fully formed sentences (delimited by . ! ?), preserving decimal values (e.g., 3.5).
    """
    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str) -> list[str]:
        sentences = []
        for c in chunk:
            has_pending_dot = (len(self.buffer) >= 2 and self.buffer[-1] == "." and self.buffer[-2].isdigit())
            if has_pending_dot:
                if c.isdigit():
                    self.buffer += c
                else:
                    cleaned = " ".join(self.buffer.split()).strip()
                    if cleaned:
                        sentences.append(cleaned)
                    self.buffer = ""
                    if c in ("!", "?"):
                        sentences.append(c)
                    elif c == ".":
                        sentences.append(".")
                    else:
                        self.buffer = c
            else:
                if c in ("!", "?"):
                    self.buffer += c
                    cleaned = " ".join(self.buffer.split()).strip()
                    if cleaned:
                        sentences.append(cleaned)
                    self.buffer = ""
                elif c == ".":
                    if len(self.buffer) >= 1 and self.buffer[-1].isdigit():
                        self.buffer += c
                    else:
                        self.buffer += c
                        cleaned = " ".join(self.buffer.split()).strip()
                        if cleaned:
                            sentences.append(cleaned)
                        self.buffer = ""
                else:
                    self.buffer += c
        return sentences

    def finish(self) -> str | None:
        tail = self.buffer
        self.buffer = ""
        cleaned = " ".join(tail.split()).strip()
        return cleaned if cleaned else None


def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành câu tại ranh giới . ! ?

    Giữ nguyên dấu câu. Không tách tại dấu phẩy.
    Số thập phân như 3.5 bên trong câu (không có khoảng trắng theo sau)
    được giữ nguyên như một đoạn.
    """
    assembler = StreamingSentenceAssembler()
    sentences = assembler.feed(text)
    tail = assembler.finish()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def discover_bleats(bleats_dir: Path) -> list[Path]:
    """Quét *bleats_dir* tìm file .wav. Trả về danh sách rỗng nếu thiếu."""
    if not bleats_dir.is_dir():
        logger.debug("Bleats directory not found: %s", bleats_dir)
        return []

    wavs = sorted(bleats_dir.glob("*.wav"))
    if not wavs:
        logger.debug("No WAV files in %s", bleats_dir)
    else:
        logger.info("Discovered %d bleat(s): %s", len(wavs), [w.name for w in wavs])

    return wavs


def load_bleat_segments(bleats_dir: Path) -> tuple[AudioSegment, ...]:
    """Quét và tải tất cả các file WAV tiếng cừu hợp lệ dưới dạng AudioSegment,
    được bình thường hóa, áp dụng fade và tăng âm.
    """
    wav_paths = discover_bleats(bleats_dir)
    segments = []
    for path in wav_paths:
        try:
            segment = AudioSegment.from_wav(str(path))
            segment = normalize_segment(segment)
            segment = segment.fade_in(BLEAT_FADE_IN_MS).fade_out(BLEAT_FADE_OUT_MS)
            segment = segment + BLEAT_VOLUME_DB
            segments.append(segment)
        except Exception:
            logger.warning("Failed to load bleat %s, skipping.", path, exc_info=True)
    return tuple(segments)


def build_inter_sentence_segment(
    bleats: tuple[AudioSegment, ...],
    *,
    probability: float = BLEAT_PROBABILITY,
    rng=random,
) -> AudioSegment:
    """Tạo một đoạn âm thanh xen giữa các câu.

    Có xác suất chèn tiếng cừu ngẫu nhiên nếu danh sách bleats không rỗng.
    Nếu không chèn, trả về khoảng lặng dài SILENCE_MS.
    """
    if bleats and rng.random() < probability:
        bleat = rng.choice(bleats)
        before = AudioSegment.silent(
            duration=PAUSE_BEFORE_BLEAT_MS,
            frame_rate=TARGET_SAMPLE_RATE,
        )
        after = AudioSegment.silent(
            duration=PAUSE_AFTER_BLEAT_MS,
            frame_rate=TARGET_SAMPLE_RATE,
        )
        before = normalize_segment(before)
        after = normalize_segment(after)
        return before + bleat + after
    else:
        silence = AudioSegment.silent(
            duration=SILENCE_MS,
            frame_rate=TARGET_SAMPLE_RATE,
        )
        return normalize_segment(silence)


# ---------------------------------------------------------------------------
# Helper audio
# ---------------------------------------------------------------------------

def normalize_segment(
    segment: AudioSegment,
    target_rate: int = TARGET_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
    target_sample_width: int = TARGET_SAMPLE_WIDTH,
) -> AudioSegment:
    """Chuyển đổi AudioSegment sang sample rate, số kênh, độ rộng chung."""
    if segment.frame_rate != target_rate:
        segment = segment.set_frame_rate(target_rate)
    if segment.channels != target_channels:
        segment = segment.set_channels(target_channels)
    if segment.sample_width != target_sample_width:
        segment = segment.set_sample_width(target_sample_width)
    return segment


def numpy_to_segment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Chuyển đổi mảng numpy float32 sang AudioSegment pydub (16-bit mono)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    return segment


# ---------------------------------------------------------------------------
# Tổng hợp TTS
# ---------------------------------------------------------------------------

class KokoroTTS:
    """Adapter ứng dụng nhỏ quanh một model ``KokoroVietnamese`` đã tải."""

    def __init__(self, engine, speed: float = SPEAKING_SPEED):
        self.engine = engine
        self.speed = speed

    def synthesize(self, text: str, output_path: str | Path | None = None):
        """Tổng hợp văn bản; tùy chọn ghi file WAV mono 24 kHz."""
        audio_array, phonemes = self.engine.synthesize(text, speed=self.speed)
        if output_path is not None:
            audio = np.clip(audio_array, -1.0, 1.0)
            pcm = (audio * 32767).astype(np.int16)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(KOKORO_SAMPLE_RATE)
                wav_file.writeframes(pcm.tobytes())
            return None
        return audio_array, phonemes


def synthesize_sentence(tts, sentence: str) -> Optional[AudioSegment]:
    """Tổng hợp một câu đơn lẻ thành AudioSegment (mono, 16-bit, 24 kHz).

    Trả về None nếu tổng hợp trống hoặc thất bại.
    """
    audio_result = tts.synthesize(sentence)
    if audio_result is None:
        return None
    audio_array, _phonemes = audio_result
    if len(audio_array) == 0:
        return None

    segment = numpy_to_segment(audio_array, KOKORO_SAMPLE_RATE)
    return normalize_segment(segment)


def synthesize_sentences(tts, sentences: list[str]) -> list[AudioSegment]:
    """Gọi Kokoro TTS cho từng câu. Trả về danh sách AudioSegments."""
    segments: list[AudioSegment] = []

    for i, sentence in enumerate(sentences):
        logger.info("Synthesizing sentence %d/%d: %.60s...", i + 1, len(sentences), sentence)
        segment = synthesize_sentence(tts, sentence)
        if segment is None:
            continue
        segments.append(segment)

    return segments


# ---------------------------------------------------------------------------
# Điều phối cấp cao nhất
# ---------------------------------------------------------------------------

def create_spoken_response(
    response_text: str,
    tts,
    bleats_dir: Path = Path(DEFAULT_BLEATS_DIR),
    runtime_dir: Path = Path(DEFAULT_RUNTIME_DIR),
) -> Path:
    """Tạo phản ứng WAV có thể nói với tiếng cừu tùy chọn.

    Raises ValueError nếu *response_text* không tạo ra câu nào có thể sử dụng.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)

    sentences = split_sentences(response_text)
    if not sentences:
        raise ValueError("Response text produced no usable sentences.")

    logger.info("Split into %d sentence(s).", len(sentences))

    sentence_segments = synthesize_sentences(tts, sentences)
    if not sentence_segments:
        raise ValueError("TTS produced no audio for any sentence.")

    bleats = load_bleat_segments(bleats_dir)

    final_audio = AudioSegment.empty()
    for i, seg in enumerate(sentence_segments):
        if i > 0:
            interstitial = build_inter_sentence_segment(bleats)
            final_audio += interstitial
        final_audio += seg

    final_path = runtime_dir / DEFAULT_FINAL_WAV
    final_audio.export(str(final_path), format="wav")
    logger.info("Exported final audio to %s (%d ms).", final_path, len(final_audio))

    return final_path
