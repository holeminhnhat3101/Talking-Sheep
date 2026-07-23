import pyaudio
import wave
from pathlib import Path
import os


class AudioPlayer:
    """Play WAV files through speaker."""
    
    def __init__(self, device_index: int | None = None):
        self.audio = pyaudio.PyAudio()
        configured = os.environ.get("AUDIO_OUTPUT_DEVICE")
        self.device_index = device_index if device_index is not None else (int(configured) if configured else None)
        
    def play_blocking(self, audio_path: str) -> None:
        """
        Play audio file and block until completion.
        
        Args:
            audio_path: Path to WAV file
        """
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 24000:
                raise ValueError("runtime/final.wav must be mono, 16-bit, and 24 kHz")
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
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
            finally:
                stream.stop_stream()
                stream.close()

    def close(self) -> None:
        if hasattr(self, "audio"):
            self.audio.terminate()
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.close()
