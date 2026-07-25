import pyaudio
import wave
from pathlib import Path

try:
    from .config import (
        AUDIO_CHUNK_SIZE,
        AUDIO_OUTPUT_DEVICE,
        TARGET_CHANNELS,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_WIDTH,
    )
except ImportError:
    from src.config import (
        AUDIO_CHUNK_SIZE,
        AUDIO_OUTPUT_DEVICE,
        TARGET_CHANNELS,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_WIDTH,
    )


class AudioPlayer:
    """Phát file WAV đồng bộ."""

    def __init__(self, device_index: int | None = None):
        self.audio = pyaudio.PyAudio()
        self.device_index = (
            device_index if device_index is not None else AUDIO_OUTPUT_DEVICE
        )
        self._closed = False

    def play_blocking(self, audio_path: str | Path) -> None:
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")

        audio_path = Path(audio_path)

        with wave.open(str(audio_path), "rb") as wav_file:
            if (
                wav_file.getnchannels() != TARGET_CHANNELS
                or wav_file.getsampwidth() != TARGET_SAMPLE_WIDTH
                or wav_file.getframerate() != TARGET_SAMPLE_RATE
            ):
                raise ValueError(
                    f"{audio_path} must be mono, 16-bit, "
                    f"{TARGET_SAMPLE_RATE} Hz"
                )

            kwargs = {
                "format": self.audio.get_format_from_width(
                    wav_file.getsampwidth()
                ),
                "channels": wav_file.getnchannels(),
                "rate": wav_file.getframerate(),
                "output": True,
            }

            if self.device_index is not None:
                kwargs["output_device_index"] = self.device_index

            stream = self.audio.open(**kwargs)
            try:
                while data := wav_file.readframes(AUDIO_CHUNK_SIZE):
                    stream.write(data)
            finally:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self.audio.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass