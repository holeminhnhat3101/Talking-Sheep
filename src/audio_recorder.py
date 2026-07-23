import logging
import array
import wave
import time
from collections import deque
from pathlib import Path
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Record audio from microphone with VAD and Raspberry Pi ALSA/PortAudio device detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
        pre_roll_duration: float = 0.25,
        min_speech_duration: float = 0.3,
        max_recording_duration: float = 15.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.device_index = device_index if device_index is not None else self._auto_detect_input_device()
        self.pre_roll_duration = pre_roll_duration
        self.min_speech_duration = min_speech_duration
        self.max_recording_duration = max_recording_duration

    def list_input_devices(self) -> list[dict]:
        """List all available audio input devices (microphones)."""
        devices = []
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = int(info.get("deviceCount", 0))

        for i in range(numdevices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if float(device_info.get("maxInputChannels", 0)) > 0:
                devices.append({
                    "index": i,
                    "name": device_info.get("name"),
                    "channels": device_info.get("maxInputChannels"),
                    "default_sample_rate": device_info.get("defaultSampleRate"),
                })
        return devices

    def _auto_detect_input_device(self) -> Optional[int]:
        """Attempt to resolve a valid default or fallback input device index for Raspberry Pi / ALSA."""
        devices = self.list_input_devices()

        if not devices:
            logger.warning("No microphone input devices detected.")
            return None

        if len(devices) == 1:
            logger.info("Only one microphone detected. Auto-selecting [%d]: %s", devices[0]["index"], devices[0]["name"])
            return devices[0]["index"]

        import sys
        if sys.stdin and sys.stdin.isatty():
            print("\nMultiple microphones detected. Please choose one:")
            for i, mic in enumerate(devices):
                print(f"  [{i + 1}] {mic['name']} (Index: {mic['index']}, Channels: {mic['channels']})")
            
            while True:
                try:
                    choice = input("Enter the number of your choice (or press Enter for default): ").strip()
                    if not choice:
                        break
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(devices):
                        selected = devices[choice_idx]
                        logger.info("User selected microphone [%d]: %s", selected["index"], selected["name"])
                        return selected["index"]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
                    
        # Fallback to default if not interactive or user pressed Enter
        try:
            default_info = self.audio.get_default_input_device_info()
            raw_idx = default_info.get("index")
            idx = int(raw_idx) if raw_idx is not None else None
            logger.info("Using default audio input device [%d]: %s", idx, default_info.get("name"))
            return idx
        except Exception:
            logger.warning("No default input device found by PortAudio. Searching available devices...")
            first_mic = devices[0]
            logger.info("Auto-selected fallback input device [%d]: %s", first_mic["index"], first_mic["name"])
            return first_mic["index"]

    def capture_utterance(
        self,
        output_path: Optional[Path] = None,
        silence_threshold: int = 500,
        silence_duration: float = 1.0,
        device_index: Optional[int] = None,
    ) -> Optional[Path]:
        """Record one utterance using four-state speech detection."""
        if output_path is None:
            output_path = Path("runtime/input.wav")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_device = device_index if device_index is not None else self.device_index

        open_kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.channels,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": self.chunk_size,
        }

        if target_device is not None:
            open_kwargs["input_device_index"] = target_device

        try:
            stream = self.audio.open(**open_kwargs)
        except Exception as exc:
            logger.error("Failed to open audio input stream on device %s. Available devices: %s", target_device, self.list_input_devices())
            raise RuntimeError(f"Microphone input stream failed to open (device index={target_device}). Check USB microphone / ALSA connection.") from exc

        frames = []
        pre_roll = deque(maxlen=max(1, int(self.pre_roll_duration * self.sample_rate / self.chunk_size)))
        state = "WAITING_FOR_SPEECH"
        consecutive_speech = 0
        silence_frames = 0
        started_at = time.monotonic()
        max_silence_frames = int(silence_duration * self.sample_rate / self.chunk_size)

        print("Recording... (speak now)")

        try:
            while True:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except IOError as exc:
                    logger.warning("Audio input overflow/read error: %s", exc)
                    continue

                rms = self._calculate_rms(bytearray(data))
                if state == "WAITING_FOR_SPEECH":
                    pre_roll.append(data)
                    consecutive_speech = consecutive_speech + 1 if rms >= silence_threshold else 0
                    if consecutive_speech >= 3:
                        state = "RECORDING"
                        frames.extend(pre_roll)
                        pre_roll.clear()
                        started_at = time.monotonic()
                else:
                    frames.append(data)
                    if rms < silence_threshold:
                        silence_frames += 1
                        if state == "RECORDING":
                            state = "TRAILING_SILENCE"
                        if state == "TRAILING_SILENCE" and silence_frames > max_silence_frames:
                            state = "COMPLETE"
                            break
                    else:
                        silence_frames = 0
                        state = "RECORDING"

                    if time.monotonic() - started_at >= self.max_recording_duration:
                        state = "COMPLETE"
                        break

            if state == "WAITING_FOR_SPEECH":
                logger.info("No speech detected.")
                return None

            if len(frames) * self.chunk_size / self.sample_rate < self.min_speech_duration:
                logger.info("Speech was shorter than the minimum duration.")
                return None
        finally:
            try:
                stream.stop_stream()
            finally:
                stream.close()

        # Save to WAV file
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        logger.info("Saved recording to %s (%d bytes)", output_path, len(b''.join(frames)))
        return output_path

    def _calculate_rms(self, data: bytearray) -> float:
        """Calculate RMS energy of audio data."""
        samples = array.array('h', data)
        sum_squares = sum(s * s for s in samples)
        return (sum_squares / len(samples)) ** 0.5 if samples else 0

    def close(self):
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except Exception:
                pass

    def __del__(self):
        self.close()
