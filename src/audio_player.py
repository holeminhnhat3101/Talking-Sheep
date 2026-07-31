import pyaudio
import time
import wave
from pathlib import Path
from pydub import AudioSegment

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
    """Phát file WAV và AudioSegment đồng bộ."""

    def __init__(self, device_index: int | None = None):
        self.audio = pyaudio.PyAudio()
        self.device_index = (
            device_index if device_index is not None else AUDIO_OUTPUT_DEVICE
        )
        self._closed = False
        self._stream = None
        self._stream_format = None
        self._in_streaming_session = False

    def _ensure_stream(self, sample_width: int, channels: int, frame_rate: int):
        fmt_key = (sample_width, channels, frame_rate, self.device_index)
        if self._stream is not None:
            if self._stream_format == fmt_key:
                return self._stream
            self._close_stream()

        kwargs = {
            "format": self.audio.get_format_from_width(sample_width),
            "channels": channels,
            "rate": frame_rate,
            "output": True,
        }
        if self.device_index is not None:
            kwargs["output_device_index"] = self.device_index

        self._stream = self.audio.open(**kwargs)
        self._stream_format = fmt_key
        return self._stream

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._stream_format = None

    def start_streaming_session(self) -> None:
        """Start a streaming session with one persistent output stream."""
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")
        
        if self._in_streaming_session:
            return  # Already in a session
        
        # Open stream with target format (48kHz, stereo, 16-bit)
        self._ensure_stream(
            TARGET_SAMPLE_WIDTH, TARGET_CHANNELS, TARGET_SAMPLE_RATE
        )
        self._in_streaming_session = True

    def end_streaming_session(self) -> None:
        """End the streaming session and drain queued PCM."""
        if not self._in_streaming_session:
            return
        
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                # Explicitly drain remaining buffered data for Bluetooth/PipeWire
                # Use a small delay to allow Bluetooth stack to process
                time.sleep(0.05)  # 50ms drain time for Bluetooth
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._stream_format = None
        
        self._in_streaming_session = False

    def write_pcm_blocking(self, pcm_data: bytes) -> None:
        """Write raw PCM data to the streaming session.
        
        Requires start_streaming_session() to be called first.
        """
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")
        
        if not self._in_streaming_session:
            raise RuntimeError("Not in streaming session. Call start_streaming_session() first.")
        
        if self._stream is None:
            raise RuntimeError("No active stream in streaming session")
        
        frame_size = TARGET_CHANNELS * TARGET_SAMPLE_WIDTH
        chunk_bytes = AUDIO_CHUNK_SIZE * frame_size
        offset = 0
        
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + chunk_bytes]
            self._stream.write(chunk)
            offset += chunk_bytes

    def play_segment_blocking(self, segment: AudioSegment) -> None:
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")

        if (
            segment.channels != TARGET_CHANNELS
            or segment.sample_width != TARGET_SAMPLE_WIDTH
            or segment.frame_rate != TARGET_SAMPLE_RATE
        ):
            raise ValueError(
                f"AudioSegment format must be stereo, 16-bit, {TARGET_SAMPLE_RATE} Hz"
            )

        stream = self._ensure_stream(
            segment.sample_width, segment.channels, segment.frame_rate
        )

        try:
            frame_size = segment.channels * segment.sample_width
            chunk_bytes = AUDIO_CHUNK_SIZE * frame_size
            data = segment.raw_data
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + chunk_bytes]
                stream.write(chunk)
                offset += chunk_bytes
        except Exception:
            self._close_stream()
            raise

    def play_blocking(self, audio_path: str | Path) -> None:
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")

        audio_path = Path(audio_path)

        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()

            if (
                channels != TARGET_CHANNELS
                or sample_width != TARGET_SAMPLE_WIDTH
                or frame_rate != TARGET_SAMPLE_RATE
            ):
                raise ValueError(
                    f"{audio_path} must be stereo, 16-bit, "
                    f"{TARGET_SAMPLE_RATE} Hz"
                )

            stream = self._ensure_stream(sample_width, channels, frame_rate)
            try:
                while data := wav_file.readframes(AUDIO_CHUNK_SIZE):
                    stream.write(data)
            except Exception:
                self._close_stream()
                raise

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._in_streaming_session = False
        try:
            self._close_stream()
        finally:
            self.audio.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass