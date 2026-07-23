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
    """Play WAV files through speaker."""
    
    def __init__(self, device_index: int | None = None):
        self.audio = pyaudio.PyAudio()
        self.device_index = (
            device_index if device_index is not None else AUDIO_OUTPUT_DEVICE
        )
        
    def play_blocking(self, audio_path: str) -> None:
        """
        Play audio file and block until completion.
        
        Args:
            audio_path: Path to WAV file
        """
        with wave.open(audio_path, 'rb') as wf:
            if (
                wf.getnchannels() != TARGET_CHANNELS
                or wf.getsampwidth() != TARGET_SAMPLE_WIDTH
                or wf.getframerate() != TARGET_SAMPLE_RATE
            ):
                raise ValueError(
                    f"runtime/final.wav must be mono, 16-bit, and "
                    f"{TARGET_SAMPLE_RATE // 1000} kHz"
                )
            kwargs = {
                "format": self.audio.get_format_from_width(wf.getsampwidth()),
                "channels": wf.getnchannels(),
                "rate": wf.getframerate(),
                "output": True,
            }
            if self.device_index is not None:
                kwargs["output_device_index"] = self.device_index
            stream = self.audio.open(**kwargs)
            try:
                data = wf.readframes(AUDIO_CHUNK_SIZE)
                while data:
                    stream.write(data)
                    data = wf.readframes(AUDIO_CHUNK_SIZE)
            finally:
                stream.stop_stream()
                stream.close()

    def close(self) -> None:
        if hasattr(self, "audio"):
            self.audio.terminate()
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.close()
