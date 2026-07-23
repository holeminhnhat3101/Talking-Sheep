"""Voice layer: sentence splitting, TTS synthesis, bleat insertion, audio composition."""

import io
import logging
import re
import secrets
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
        TARGET_CHANNELS,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_WIDTH,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries.

    Preserves punctuation. Does not split on commas.
    Decimal numbers like 3.5 inside sentences (without following whitespace)
    are preserved as single segments.
    """
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Bleat discovery and selection
# ---------------------------------------------------------------------------

def discover_bleats(bleats_dir: Path) -> list[Path]:
    """Scan *bleats_dir* for .wav files.  Return empty list if missing."""
    if not bleats_dir.is_dir():
        logger.debug("Bleats directory not found: %s", bleats_dir)
        return []

    wavs = sorted(bleats_dir.glob("*.wav"))
    if not wavs:
        logger.debug("No WAV files in %s", bleats_dir)
    else:
        logger.info("Discovered %d bleat(s): %s", len(wavs), [w.name for w in wavs])

    return wavs


def choose_bleat(bleats: list[Path]) -> Optional[Path]:
    """Pick one bleat file from the available list.  Return None if empty."""
    if not bleats:
        return None

    return secrets.choice(bleats)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def normalize_segment(
    segment: AudioSegment,
    target_rate: int = TARGET_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
    target_sample_width: int = TARGET_SAMPLE_WIDTH,
) -> AudioSegment:
    """Convert an AudioSegment to a common sample rate, channel count, width."""
    if segment.frame_rate != target_rate:
        segment = segment.set_frame_rate(target_rate)
    if segment.channels != target_channels:
        segment = segment.set_channels(target_channels)
    if segment.sample_width != target_sample_width:
        segment = segment.set_sample_width(target_sample_width)
    return segment


def numpy_to_segment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a float32 numpy array to a pydub AudioSegment (16-bit mono)."""
    # Clip and scale float32 [-1, 1] to int16
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
# TTS synthesis
# ---------------------------------------------------------------------------

class KokoroTTS:
    """Small application adapter around one loaded ``KokoroVietnamese`` model."""

    def __init__(self, engine, speed: float = SPEAKING_SPEED):
        self.engine = engine
        self.speed = speed

    def synthesize(self, text: str, output_path: str | Path | None = None):
        """Synthesize text; optionally write a mono 24 kHz WAV file."""
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

def synthesize_sentences(tts, sentences: list[str]) -> list[AudioSegment]:
    """Call Kokoro TTS for each sentence.  Returns list of AudioSegments.

    ``tts`` must have a ``synthesize(text) -> (np.ndarray, str)`` method
    (the KokoroVietnamese API).
    """
    segments: list[AudioSegment] = []

    for i, sentence in enumerate(sentences):
        logger.info("Synthesizing sentence %d/%d: %.60s...", i + 1, len(sentences), sentence)

        audio_result = tts.synthesize(sentence)
        if audio_result is None:
            raise RuntimeError("TTS adapter returned no in-memory audio for composition")
        audio_array, _phonemes = audio_result

        if len(audio_array) == 0:
            logger.warning("TTS returned empty audio for sentence %d, skipping.", i + 1)
            continue

        segment = numpy_to_segment(audio_array, KOKORO_SAMPLE_RATE)
        segment = normalize_segment(segment)
        segments.append(segment)

    return segments


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_with_bleat(
    sentence_segments: list[AudioSegment],
    bleat_path: Optional[Path],
    bleat_after_index: int = 0,
) -> AudioSegment:
    """Combine sentence AudioSegments with optional single bleat.

    At most one bleat is inserted.  The bleat goes after
    ``sentence_segments[bleat_after_index]`` (default: after the first sentence).
    If there are fewer than 2 sentences, no bleat is inserted.
    """
    if not sentence_segments:
        return AudioSegment.empty()

    pause = AudioSegment.silent(duration=SILENCE_MS, frame_rate=TARGET_SAMPLE_RATE)
    result = AudioSegment.empty()

    insert_bleat = (
        bleat_path is not None
        and bleat_path.is_file()
        and len(sentence_segments) >= 2
    )

    bleat_segment = None
    if insert_bleat:
        try:
            bleat_segment = AudioSegment.from_wav(str(bleat_path))
            bleat_segment = normalize_segment(bleat_segment)
            bleat_segment = bleat_segment.fade_in(BLEAT_FADE_IN_MS).fade_out(BLEAT_FADE_OUT_MS)
            bleat_segment = bleat_segment + BLEAT_VOLUME_DB
        except Exception:
            logger.warning("Failed to load bleat %s, continuing without.", bleat_path, exc_info=True)
            bleat_segment = None

    for i, seg in enumerate(sentence_segments):
        if i > 0:
            result += pause

        result += seg

        if bleat_segment is not None and i == bleat_after_index:
            before = AudioSegment.silent(duration=PAUSE_BEFORE_BLEAT_MS, frame_rate=TARGET_SAMPLE_RATE)
            after = AudioSegment.silent(duration=PAUSE_AFTER_BLEAT_MS, frame_rate=TARGET_SAMPLE_RATE)
            result += before + bleat_segment + after

    return result


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def create_spoken_response(
    response_text: str,
    tts,
    bleats_dir: Path = Path(DEFAULT_BLEATS_DIR),
    runtime_dir: Path = Path(DEFAULT_RUNTIME_DIR),
) -> Path:
    """Create a spoken WAV response with optional sheep bleat.

    Steps:
      1. Split response into sentences.
      2. Synthesize each sentence via Kokoro TTS.
      3. Discover available bleats, choose one.
      4. Compose final audio with optional bleat after the first sentence.
      5. Export to runtime_dir/{DEFAULT_FINAL_WAV} and return its path.

    Raises ValueError if *response_text* produces no usable sentences.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)

    sentences = split_sentences(response_text)
    if not sentences:
        raise ValueError("Response text produced no usable sentences.")

    logger.info("Split into %d sentence(s).", len(sentences))

    # Synthesize
    sentence_segments = synthesize_sentences(tts, sentences)
    if not sentence_segments:
        raise ValueError("TTS produced no audio for any sentence.")

    # Bleat
    bleats = discover_bleats(bleats_dir)
    bleat_path = choose_bleat(bleats) if len(sentences) >= 2 and random.random() < BLEAT_PROBABILITY else None

    # Compose
    final_audio = compose_with_bleat(
        sentence_segments=sentence_segments,
        bleat_path=bleat_path,
        bleat_after_index=0,
    )

    # Export
    final_path = runtime_dir / DEFAULT_FINAL_WAV
    final_audio.export(str(final_path), format="wav")
    logger.info("Exported final audio to %s (%d ms).", final_path, len(final_audio))

    return final_path
