"""Microphone capture with VAD and streaming 16 kHz audio delivery."""

from __future__ import annotations

import logging
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyaudio

try:
    from .config import (
        AUDIO_AUTO_CALIBRATE,
        AUDIO_CALIBRATION_DURATION,
        AUDIO_CAPTURE_CHANNELS,
        AUDIO_CAPTURE_RATE,
        AUDIO_CHANNEL_MODE,
        AUDIO_CHUNK_SIZE,
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        DEFAULT_INPUT_WAV,
        DEFAULT_RUNTIME_DIR,
        MAX_RECORDING_DURATION,
        MIN_SPEECH_DURATION,
        PRE_ROLL_DURATION,
        SILENCE_DURATION,
        SILENCE_THRESHOLD,
        STT_CHANNELS,
        STT_SAMPLE_RATE,
    )
except ImportError:
    from src.config import (
        AUDIO_AUTO_CALIBRATE,
        AUDIO_CALIBRATION_DURATION,
        AUDIO_CAPTURE_CHANNELS,
        AUDIO_CAPTURE_RATE,
        AUDIO_CHANNEL_MODE,
        AUDIO_CHUNK_SIZE,
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        DEFAULT_INPUT_WAV,
        DEFAULT_RUNTIME_DIR,
        MAX_RECORDING_DURATION,
        MIN_SPEECH_DURATION,
        PRE_ROLL_DURATION,
        SILENCE_DURATION,
        SILENCE_THRESHOLD,
        STT_CHANNELS,
        STT_SAMPLE_RATE,
    )

logger = logging.getLogger(__name__)
AudioChunkCallback = Callable[[np.ndarray], None]


class MicrophoneUnavailableError(RuntimeError):
    """Raised when no usable microphone stream can be opened."""


class AudioRecorder:
    """Capture native microphone audio and stream 16 kHz mono float32 to STT."""

    def __init__(
        self,
        sample_rate: int = STT_SAMPLE_RATE,
        channels: int = STT_CHANNELS,
        chunk_size: int = AUDIO_CHUNK_SIZE,
        device_index: int | str | None = None,
        pre_roll_duration: float = PRE_ROLL_DURATION,
        min_speech_duration: float = MIN_SPEECH_DURATION,
        max_recording_duration: float = MAX_RECORDING_DURATION,
        capture_rate: int | None = AUDIO_CAPTURE_RATE,
        capture_channels: int | None = AUDIO_CAPTURE_CHANNELS,
        channel_mode: str = AUDIO_CHANNEL_MODE,
        auto_calibrate: bool = AUDIO_AUTO_CALIBRATE,
        calibration_duration: float = AUDIO_CALIBRATION_DURATION,
        threshold_multiplier: float = AUDIO_THRESHOLD_MULTIPLIER,
        minimum_auto_threshold: float = AUDIO_MINIMUM_AUTO_THRESHOLD,
        speech_start_chunks: int = AUDIO_SPEECH_START_CHUNKS,
        max_wait_for_speech: float | None = AUDIO_MAX_WAIT_FOR_SPEECH,
        save_native_debug: bool = AUDIO_SAVE_NATIVE_DEBUG,
    ) -> None:
        if sample_rate <= 0 or channels != 1 or chunk_size <= 0:
            raise ValueError(
                "STT output must be mono with a positive rate and chunk size"
            )

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.device_selector = device_index
        self.pre_roll_duration = float(pre_roll_duration)
        self.min_speech_duration = float(min_speech_duration)
        self.max_recording_duration = float(max_recording_duration)
        self.capture_rate = capture_rate
        self.capture_channels = capture_channels
        self.channel_mode = channel_mode.strip().lower()
        self.auto_calibrate = bool(auto_calibrate)
        self.calibration_duration = float(calibration_duration)
        self.threshold_multiplier = float(threshold_multiplier)
        self.minimum_auto_threshold = float(minimum_auto_threshold)
        self.speech_start_chunks = max(1, int(speech_start_chunks))
        self.max_wait_for_speech = max_wait_for_speech
        self.save_native_debug = bool(save_native_debug)

        self.audio = pyaudio.PyAudio()
        self._closed = False
        self._cached_config: dict | None = None
        self.last_capture_info: dict = {}
        self._pending_debug: dict | None = None

        self._resample_input_rate = 0
        self._resample_step = 1.0
        self._resample_position = 0.0
        self._resample_tail = np.empty(0, dtype=np.float32)

    def list_input_devices(self) -> list[dict]:
        devices = []
        for index in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(index)
            except Exception:
                continue

            channels = int(info.get("maxInputChannels", 0) or 0)
            if channels < 1:
                continue

            host_api_index = int(info.get("hostApi", 0) or 0)
            try:
                host_api = self.audio.get_host_api_info_by_index(
                    host_api_index
                ).get("name", str(host_api_index))
            except Exception:
                host_api = str(host_api_index)

            devices.append(
                {
                    "index": index,
                    "name": str(info.get("name", f"Input device {index}")),
                    "channels": channels,
                    "default_sample_rate": int(
                        round(float(info.get("defaultSampleRate", self.sample_rate)))
                    ),
                    "host_api": str(host_api),
                }
            )
        return devices

    def list_selectable_input_devices(self) -> list[dict]:
        return [
            device
            for device in self.list_input_devices()
            if not self._is_virtual_device(device)
        ]

    @staticmethod
    def _is_virtual_device(device: dict) -> bool:
        """Check if a device name suggests it is a virtual/pseudo ALSA device."""
        name = str(device["name"]).casefold()
        virtual_names = (
            "default",
            "sysdefault",
            "pulse",
            "pipewire",
            "jack",
            "dmix",
            "dsnoop",
            "surround",
            "front",
            "rear",
            "center_lfe",
            "iec958",
            "spdif",
            "modem",
            "phoneline",
        )
        return any(token in name for token in virtual_names)

    @staticmethod
    def _is_respeaker(device: dict) -> bool:
        """Check if a device name suggests it is a Seeed Studio ReSpeaker."""
        name = str(device["name"]).casefold()
        return "respeaker" in name or "seeed" in name

    @classmethod
    def _device_score(cls, device: dict) -> int:
        """Calculate a quality score for a device to aid auto-selection."""
        name = str(device["name"]).casefold()
        channels = int(device.get("channels", 0) or 0)

        if channels < 1:
            return -10000

        # Penalize virtual devices heavily
        if cls._is_virtual_device(device):
            return -1000

        score = 0

        # Highly prefer ReSpeaker
        if cls._is_respeaker(device):
            score += 1000

        # Prefer hardware devices (hw: or plughw:)
        if "(hw:" in name or "(plughw:" in name:
            score += 200

        # Prefer USB devices
        if "usb" in name:
            score += 100

        # Penalize suspiciously high channel counts for non-Respeaker
        if channels > 32 and not cls._is_respeaker(device):
            score -= 500

        return score

    def _select_device(self) -> dict:
        devices = self.list_input_devices()
        if not devices:
            raise MicrophoneUnavailableError("No microphone input devices detected")

        selector = self.device_selector
        if isinstance(selector, int):
            match = next((d for d in devices if d["index"] == selector), None)
            if match:
                return match
            raise MicrophoneUnavailableError(
                f"Microphone index {selector} is unavailable"
            )

        if isinstance(selector, str) and selector.strip():
            wanted = selector.casefold().strip()
            exact = [d for d in devices if d["name"].casefold() == wanted]
            partial = [d for d in devices if wanted in d["name"].casefold()]
            if exact or partial:
                return (exact or partial)[0]
            raise MicrophoneUnavailableError(
                f"Microphone {selector!r} is unavailable"
            )

        # Automatic selection logic
        physical_devices = [d for d in devices if not self._is_virtual_device(d)]

        if physical_devices:
            best_device = max(physical_devices, key=self._device_score)
            # Guardrail: even if it's the "best" physical device, if it has 128 channels,
            # it's likely a misidentified virtual device.
            if int(best_device["channels"]) > 32:
                raise MicrophoneUnavailableError(
                    f"Auto-selected microphone appears to be a virtual ALSA device: "
                    f"{best_device['name']} ({best_device['channels']} channels). "
                    "Use --input-device to select a physical microphone."
                )
            return best_device

        # Fallback to PortAudio default
        try:
            default_index = int(self.audio.get_default_input_device_info()["index"])
            match = next(
                (d for d in devices if d["index"] == default_index),
                None,
            )
            if match:
                logger.warning(
                    "No physical microphone candidate found; "
                    "falling back to PortAudio default: %s",
                    match["name"],
                )
                return match
        except Exception:
            pass

        return devices[0]

    def _candidate_configs(self, device: dict) -> list[dict]:
        max_channels = int(device["channels"])
        is_respeaker = self._is_respeaker(device)

        # Standard negotiation rates and channels
        rates = self._unique(
            [
                *([self.capture_rate] if self.capture_rate else []),
                int(device["default_sample_rate"]),
                48000,
                44100,
                32000,
                self.sample_rate,
            ]
        )
        channel_counts = self._unique(
            [
                *([self.capture_channels] if self.capture_channels else []),
                max_channels,
                1,
                min(2, max_channels),
            ]
        )

        generic_configs = [
            {
                "device_index": int(device["index"]),
                "rate": rate,
                "channels": channels,
                "format": audio_format,
            }
            for rate in rates
            for channels in channel_counts
            for audio_format in (pyaudio.paInt16, pyaudio.paFloat32)
            if channels <= max_channels
        ]

        respeaker_configs = []
        if is_respeaker:
            # ReSpeaker v2.0/v2.0.1 firmware profiles
            # 6-channel firmware provides processed audio on channel 0 at 16kHz
            if max_channels >= 6:
                respeaker_configs.append(
                    {
                        "device_index": int(device["index"]),
                        "rate": 16000,
                        "channels": 6,
                        "format": pyaudio.paInt16,
                    }
                )
            # 1-channel firmware fallback
            respeaker_configs.append(
                {
                    "device_index": int(device["index"]),
                    "rate": 16000,
                    "channels": 1,
                    "format": pyaudio.paInt16,
                }
            )

        # Prioritize ReSpeaker configs, then others
        configs = respeaker_configs + [
            c for c in generic_configs if c not in respeaker_configs
        ]

        if (
            self._cached_config
            and self._cached_config["device_index"] == device["index"]
        ):
            configs = [self._cached_config] + [
                config for config in configs if config != self._cached_config
            ]
        return configs

    @staticmethod
    def _unique(values: list[int]) -> list[int]:
        return list(dict.fromkeys(int(value) for value in values if value and value > 0))

    def _open_stream(self, device: dict) -> tuple[pyaudio.Stream, dict]:
        failures = []
        for config in self._candidate_configs(device):
            try:
                stream = self.audio.open(
                    format=config["format"],
                    channels=config["channels"],
                    rate=config["rate"],
                    input=True,
                    input_device_index=config["device_index"],
                    frames_per_buffer=self.chunk_size,
                )
            except Exception as exc:
                failures.append(
                    f"{config['rate']} Hz/{config['channels']} ch: {exc}"
                )
                continue

            self._cached_config = config
            logger.info(
                "Using microphone [%d] %s at %d Hz, %d channel(s)",
                device["index"],
                device["name"],
                config["rate"],
                config["channels"],
            )
            return stream, config

        self._cached_config = None
        raise MicrophoneUnavailableError(
            f"Could not open [{device['index']}] {device['name']}. "
            f"Last attempts: {'; '.join(failures[-4:])}"
        )

    def capture_utterance_stream(
        self,
        on_audio_chunk: AudioChunkCallback,
        output_path: Optional[Path] = None,
        silence_threshold: Optional[float] = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
        device_index: int | str | None = None,
    ) -> tuple[Optional[Path], float]:
        """Stream one utterance to STT while VAD controls its boundaries."""
        if self._closed:
            raise RuntimeError("AudioRecorder is closed")
        if silence_duration <= 0:
            raise ValueError("silence_duration must be positive")

        output_path = Path(
            output_path or Path(DEFAULT_RUNTIME_DIR) / DEFAULT_INPUT_WAV
        )
        old_selector = self.device_selector
        if device_index is not None:
            self.device_selector = device_index

        self._pending_debug = None
        stream = None
        try:
            device = self._select_device()
            logger.info(
                "Selected microphone [%d] %s (host=%s, max_channels=%d, respeaker=%s)",
                device["index"],
                device["name"],
                device["host_api"],
                device["channels"],
                self._is_respeaker(device),
            )
            stream, config = self._open_stream(device)
            rate = int(config["rate"])
            channels = int(config["channels"])
            audio_format = int(config["format"])

            logger.info(
                "Using microphone [%d] %s at %d Hz, %d channel(s), mode=%s",
                device["index"],
                device["name"],
                rate,
                channels,
                (
                    "channel:0"
                    if self._is_respeaker(device)
                    and channels >= 6
                    and self.channel_mode == "auto"
                    else self.channel_mode
                ),
            )
            chunk_seconds = self.chunk_size / rate

            pre_roll: deque[bytes] = deque(
                maxlen=max(1, round(self.pre_roll_duration / chunk_seconds))
            )
            silence_chunks_needed = max(
                1, round(silence_duration / chunk_seconds)
            )

            if silence_threshold is not None:
                threshold = float(silence_threshold)
            elif self.auto_calibrate:
                threshold = self._calibrate_threshold(stream, config)
            else:
                threshold = self.minimum_auto_threshold

            native_debug_frames = [] if self.save_native_debug else None
            debug_pcm_chunks = [] if self.save_native_debug else None
            selected_channel: int | None = None
            consecutive_speech = 0
            trailing_silence = 0
            speech_chunks = 0
            recording_started: float | None = None
            wait_started = time.monotonic()
            read_errors = 0

            print("Recording... (speak now)")

            while True:
                try:
                    while stream.get_read_available() < self.chunk_size:
                        time.sleep(0.01)
                    data = stream.read(
                        self.chunk_size,
                        exception_on_overflow=False,
                    )
                    read_errors = 0
                except KeyboardInterrupt:
                    raise
                except (IOError, OSError) as exc:
                    read_errors += 1
                    if read_errors >= 3:
                        raise MicrophoneUnavailableError(
                            f"Microphone read failed repeatedly: {exc}"
                        ) from exc
                    continue

                rms = self._rms(data, audio_format, channels)

                if recording_started is None:
                    if (
                        self.max_wait_for_speech is not None
                        and time.monotonic() - wait_started
                        >= self.max_wait_for_speech
                    ):
                        logger.info(
                            "No speech detected within %.2f seconds",
                            self.max_wait_for_speech,
                        )
                        return None, 0.0

                    pre_roll.append(data)
                    consecutive_speech = (
                        consecutive_speech + 1 if rms >= threshold else 0
                    )
                    if consecutive_speech < self.speech_start_chunks:
                        continue

                    selected_channel = self._choose_channel(
                        self._decode(pre_roll[-1], audio_format, channels),
                        is_respeaker=self._is_respeaker(device),
                    )
                    self._reset_stream_resampler(rate)

                    for pre_roll_frame in pre_roll:
                        self._deliver_frame(
                            pre_roll_frame,
                            audio_format,
                            channels,
                            selected_channel,
                            on_audio_chunk,
                            debug_pcm_chunks,
                        )

                    if native_debug_frames is not None:
                        native_debug_frames.extend(pre_roll)

                    pre_roll.clear()
                    recording_started = time.monotonic()
                    speech_chunks = consecutive_speech
                    continue

                self._deliver_frame(
                    data,
                    audio_format,
                    channels,
                    selected_channel,
                    on_audio_chunk,
                    debug_pcm_chunks,
                )
                if native_debug_frames is not None:
                    native_debug_frames.append(data)

                if rms >= threshold:
                    speech_chunks += 1
                    trailing_silence = 0
                else:
                    trailing_silence += 1
                    if trailing_silence >= silence_chunks_needed:
                        break

                if (
                    time.monotonic() - recording_started
                    >= self.max_recording_duration
                ):
                    break

            speech_end_time = time.monotonic()
            speech_duration = speech_chunks * chunk_seconds
            if speech_duration < self.min_speech_duration:
                logger.info("Speech was too short: %.2f seconds", speech_duration)
                return None, 0.0

            if self.save_native_debug:
                self._pending_debug = {
                    "output_path": output_path,
                    "native_path": output_path.with_name(
                        f"{output_path.stem}_native{output_path.suffix}"
                    ),
                    "native_frames": native_debug_frames or [],
                    "pcm_chunks": debug_pcm_chunks or [],
                    "rate": rate,
                    "channels": channels,
                    "audio_format": audio_format,
                }

            self.last_capture_info = {
                "device_index": device["index"],
                "device_name": device["name"],
                "capture_rate": rate,
                "capture_channels": channels,
                "target_rate": self.sample_rate,
                "speech_duration": speech_duration,
            }
            return output_path, speech_end_time
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self.device_selector = old_selector

    def _deliver_frame(
        self,
        data: bytes,
        audio_format: int,
        channels: int,
        selected_channel: int,
        callback: AudioChunkCallback,
        debug_pcm_chunks: list[bytes] | None,
    ) -> None:
        decoded = self._decode(data, audio_format, channels)
        mono = self._select_fixed_channel(decoded, selected_channel)
        converted = self._resample_stream_chunk(mono)
        if converted.size == 0:
            return

        callback(converted)
        if debug_pcm_chunks is not None:
            debug_pcm_chunks.append(self._to_int16(converted).tobytes())

    def _choose_channel(self, samples: np.ndarray, *, is_respeaker: bool = False) -> int:
        if samples.shape[1] == 1 or self.channel_mode == "first":
            return 0
        if self.channel_mode == "mix":
            return -1

        # ReSpeaker 6-channel processed audio is on channel 0
        if is_respeaker and samples.shape[1] >= 6 and self.channel_mode == "auto":
            return 0

        if self.channel_mode.startswith("channel:"):
            index = int(self.channel_mode.split(":", 1)[1])
            if not 0 <= index < samples.shape[1]:
                raise ValueError(
                    f"channel mode {self.channel_mode!r} is out of range "
                    f"for {samples.shape[1]} channel(s)"
                )
            return index

        channel_rms = np.sqrt(
            np.mean(np.square(samples, dtype=np.float64), axis=0)
        )
        return int(np.argmax(channel_rms))

    @staticmethod
    def _select_fixed_channel(
        samples: np.ndarray,
        selected_channel: int,
    ) -> np.ndarray:
        if samples.shape[1] == 1:
            return samples[:, 0]
        if selected_channel == -1:
            return np.asarray(samples.mean(axis=1), dtype=np.float32)
        return samples[:, selected_channel]

    def _reset_stream_resampler(self, input_rate: int) -> None:
        self._resample_input_rate = int(input_rate)
        self._resample_step = input_rate / self.sample_rate
        self._resample_position = 0.0
        self._resample_tail = np.empty(0, dtype=np.float32)

    def _resample_stream_chunk(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0 or self._resample_input_rate == self.sample_rate:
            return audio

        if self._resample_tail.size:
            audio = np.concatenate((self._resample_tail, audio))
        if audio.size < 2:
            self._resample_tail = audio.copy()
            return np.empty(0, dtype=np.float32)

        limit = audio.size - 1
        positions = np.arange(
            self._resample_position,
            limit,
            self._resample_step,
            dtype=np.float64,
        )
        if positions.size:
            converted = np.interp(
                positions,
                np.arange(audio.size, dtype=np.float64),
                audio,
            ).astype(np.float32)
            next_position = float(positions[-1] + self._resample_step)
        else:
            converted = np.empty(0, dtype=np.float32)
            next_position = self._resample_position

        self._resample_position = next_position - limit
        self._resample_tail = audio[-1:].copy()
        return np.clip(converted, -1.0, 1.0)

    def _calibrate_threshold(self, stream: pyaudio.Stream, config: dict) -> float:
        values = []
        count = max(
            1,
            round(self.calibration_duration * config["rate"] / self.chunk_size),
        )
        for _ in range(count):
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except (IOError, OSError):
                continue
            values.append(self._rms(data, config["format"], config["channels"]))

        ambient = float(np.median(values)) if values else 0.0
        threshold = max(
            self.minimum_auto_threshold,
            ambient * self.threshold_multiplier,
        )
        logger.info(
            "Ambient RMS %.1f; speech threshold %.1f",
            ambient,
            threshold,
        )
        return threshold

    @staticmethod
    def _decode(data: bytes, audio_format: int, channels: int) -> np.ndarray:
        if audio_format == pyaudio.paInt16:
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif audio_format == pyaudio.paFloat32:
            samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

        usable = samples.size - samples.size % channels
        return samples[:usable].reshape(-1, channels)

    @classmethod
    def _rms(cls, data: bytes, audio_format: int, channels: int) -> float:
        samples = cls._decode(data, audio_format, channels)
        if samples.size == 0:
            return 0.0
        channel_rms = np.sqrt(
            np.mean(np.square(samples, dtype=np.float64), axis=0)
        )
        return float(np.max(channel_rms) * 32768.0)

    @staticmethod
    def _to_int16(samples: np.ndarray) -> np.ndarray:
        return np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")

    def save_pending_debug_audio(self) -> None:
        """Write optional WAV diagnostics after STT finalization."""
        pending = self._pending_debug
        self._pending_debug = None
        if pending is None:
            return

        try:
            output_path = pending["output_path"]
            output_path.parent.mkdir(parents=True, exist_ok=True)

            pcm_chunks = pending["pcm_chunks"]
            if pcm_chunks:
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(b"".join(pcm_chunks))
                logger.info("Saved debug WAV to %s", output_path)

            native_frames = pending["native_frames"]
            if native_frames:
                native = self._decode(
                    b"".join(native_frames),
                    pending["audio_format"],
                    pending["channels"],
                )
                native_path = pending["native_path"]
                with wave.open(str(native_path), "wb") as native_wav:
                    native_wav.setnchannels(pending["channels"])
                    native_wav.setsampwidth(2)
                    native_wav.setframerate(pending["rate"])
                    native_wav.writeframes(self._to_int16(native).tobytes())
                logger.info("Saved native debug WAV to %s", native_path)
        except Exception:
            logger.exception("Failed to save debug audio")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio.terminate()

    def __enter__(self) -> "AudioRecorder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass