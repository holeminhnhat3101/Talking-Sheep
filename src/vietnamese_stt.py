"""Native-streaming Vietnamese STT with sherpa-onnx Zipformer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import (
        STT_DECODER_PATH,
        STT_DECODING_METHOD,
        STT_ENCODER_PATH,
        STT_JOINER_PATH,
        STT_LOG_PARTIALS,
        STT_MAX_ACTIVE_PATHS,
        STT_NUM_THREADS,
        STT_PROVIDER,
        STT_SAMPLE_RATE,
        STT_TOKENS_PATH,
    )
except ImportError:
    from src.config import (
        STT_DECODER_PATH,
        STT_DECODING_METHOD,
        STT_ENCODER_PATH,
        STT_JOINER_PATH,
        STT_LOG_PARTIALS,
        STT_MAX_ACTIVE_PATHS,
        STT_NUM_THREADS,
        STT_PROVIDER,
        STT_SAMPLE_RATE,
        STT_TOKENS_PATH,
    )

logger = logging.getLogger(__name__)


class VietnameseSTT:
    """Persistent Zipformer recognizer with one online stream per utterance."""

    def __init__(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is missing. Install requirements-rpi.txt."
            ) from exc

        required = (
            Path(STT_ENCODER_PATH),
            Path(STT_DECODER_PATH),
            Path(STT_JOINER_PATH),
            Path(STT_TOKENS_PATH),
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Missing Zipformer model files: " + ", ".join(missing)
            )

        logger.info("Loading streaming Zipformer STT...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(STT_TOKENS_PATH),
            encoder=str(STT_ENCODER_PATH),
            decoder=str(STT_DECODER_PATH),
            joiner=str(STT_JOINER_PATH),
            num_threads=STT_NUM_THREADS,
            provider=STT_PROVIDER,
            sample_rate=STT_SAMPLE_RATE,
            feature_dim=80,
            decoding_method=STT_DECODING_METHOD,
            max_active_paths=STT_MAX_ACTIVE_PATHS,
        )
        self.sample_rate = STT_SAMPLE_RATE
        self.stream: Any | None = None
        self._last_partial = ""

    def start_utterance(self) -> None:
        """Create a fresh decoder stream while reusing the loaded model."""
        self.stream = self.recognizer.create_stream()
        self._last_partial = ""

    def accept_audio(self, samples: np.ndarray) -> None:
        """Accept one 16 kHz mono float32 chunk and decode available frames."""
        if self.stream is None:
            raise RuntimeError("start_utterance() must be called first")

        audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return

        self.stream.accept_waveform(self.sample_rate, audio)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        if STT_LOG_PARTIALS:
            partial = self._result_text()
            if partial and partial != self._last_partial:
                logger.debug("STT partial: %s", partial)
                self._last_partial = partial

    def finish_utterance(self, speech_end_time: float) -> str:
        """Flush the current stream and return its final transcript."""
        if self.stream is None:
            return ""

        try:
            # sherpa-onnx examples add a short zero tail before input_finished()
            # so the transducer can emit tokens for the final speech frames.
            tail = np.zeros(int(0.66 * self.sample_rate), dtype=np.float32)
            self.stream.accept_waveform(self.sample_rate, tail)
            self.stream.input_finished()

            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            transcript = " ".join(self._result_text().split())
            latency_ms = (time.perf_counter() - speech_end_time) * 1000
            logger.info("VAD end to transcript: %.1f ms", latency_ms)
            return transcript
        finally:
            self.stream = None
            self._last_partial = ""

    def abort_utterance(self) -> None:
        """Discard an incomplete utterance without reloading the model."""
        self.stream = None
        self._last_partial = ""

    def _result_text(self) -> str:
        if self.stream is None:
            return ""

        result = self.recognizer.get_result(self.stream)
        text = getattr(result, "text", result if isinstance(result, str) else "")
        return str(text).strip()