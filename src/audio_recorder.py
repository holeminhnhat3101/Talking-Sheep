from __future__ import annotations

import logging
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio

try:
    from .config import (
        AUDIO_AUTO_CALIBRATE,
        AUDIO_CALIBRATION_DURATION,
        AUDIO_CAPTURE_CHANNELS,
        AUDIO_CAPTURE_RATE,
        AUDIO_CHANNEL_MODE,
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        AUDIO_CHUNK_SIZE,
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
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        AUDIO_CHUNK_SIZE,
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


class MicrophoneUnavailableError(RuntimeError):
    """Được raise khi không thể mở stream microphone nào được sử dụng."""


class AudioRecorder:
    """Ghi âm từ bất kỳ microphone PortAudio nào, sau đó ghi file 16 kHz mono cho Whisper."""

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
            raise ValueError("Whisper output must be mono with a positive rate and chunk size")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_selector = device_index
        self.pre_roll_duration = pre_roll_duration
        self.min_speech_duration = min_speech_duration
        self.max_recording_duration = max_recording_duration
        self.capture_rate = capture_rate
        self.capture_channels = capture_channels
        self.channel_mode = channel_mode
        self.auto_calibrate = auto_calibrate
        self.calibration_duration = calibration_duration
        self.threshold_multiplier = threshold_multiplier
        self.minimum_auto_threshold = minimum_auto_threshold
        self.speech_start_chunks = max(1, int(speech_start_chunks))
        self.max_wait_for_speech = max_wait_for_speech
        self.save_native_debug = save_native_debug

        self.audio = pyaudio.PyAudio()
        self._closed = False
        self._cached_config: dict | None = None
        self.last_capture_info: dict = {}

    def list_input_devices(self) -> list[dict]:
        """Trả về mọi thiết bị đầu vào PortAudio trên tất cả host API."""
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
                host_api = self.audio.get_host_api_info_by_index(host_api_index).get(
                    "name", str(host_api_index)
                )
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

    def _select_device(self) -> dict:
        devices = self.list_input_devices()
        if not devices:
            raise MicrophoneUnavailableError("No microphone input devices detected")

        selector = self.device_selector

        if isinstance(selector, int):
            for device in devices:
                if device["index"] == selector:
                    return device
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

        try:
            default_index = int(self.audio.get_default_input_device_info()["index"])
            for device in devices:
                if device["index"] == default_index:
                    return device
        except Exception:
            pass

        return next(
            (d for d in devices if "usb" in d["name"].casefold()),
            devices[0],
        )

    def _candidate_configs(self, device: dict) -> list[dict]:
        max_channels = device["channels"]
        default_rate = device["default_sample_rate"]

        rates = self._unique(
            [
                *( [self.capture_rate] if self.capture_rate else [] ),
                default_rate,
                48000,
                44100,
                32000,
                self.sample_rate,
                STT_SAMPLE_RATE,
            ]
        )
        channel_counts = self._unique(
            [
                *( [self.capture_channels] if self.capture_channels else [] ),
                max_channels,
                1,
                min(2, max_channels),
            ]
        )
        formats = [pyaudio.paInt16, pyaudio.paFloat32]

        configs = [
            {
                "device_index": device["index"],
                "rate": rate,
                "channels": channels,
                "format": audio_format,
            }
            for rate in rates
            for channels in channel_counts
            for audio_format in formats
            if channels <= max_channels
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
        return list(dict.fromkeys(value for value in values if value > 0))

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

    def capture_utterance(
        self,
        output_path: Optional[Path] = None,
        silence_threshold: Optional[float] = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
        device_index: int | str | None = None,
    ) -> Optional[Path]:
        """Thu âm theo native, sau đó chuyển đổi sang WAV 16 kHz mono cho Whisper."""
        if self._closed:
            raise RuntimeError("AudioRecorder is closed")
        if silence_duration <= 0:
            raise ValueError("silence_duration must be positive")

        output_path = Path(
            output_path or Path(DEFAULT_RUNTIME_DIR) / DEFAULT_INPUT_WAV
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        old_selector = self.device_selector
        if device_index is not None:
            self.device_selector = device_index

        stream = None
        try:
            device = self._select_device()
            stream, config = self._open_stream(device)

            rate = config["rate"]
            channels = config["channels"]
            audio_format = config["format"]
            chunk_seconds = self.chunk_size / rate

            pre_roll = deque(
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

            frames: list[bytes] = []
            speech_chunks = 0
            consecutive_speech = 0
            trailing_silence = 0
            recording_started = None
            read_errors = 0
            wait_started = time.monotonic()

            print("Recording... (speak now)")

            while True:
                try:
                    # Avoid blocking indefinitely inside PortAudio so Ctrl+C works.
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
                        return None

                    pre_roll.append(data)
                    consecutive_speech = (
                        consecutive_speech + 1 if rms >= threshold else 0
                    )

                    if consecutive_speech < self.speech_start_chunks:
                        continue

                    frames.extend(pre_roll)
                    pre_roll.clear()
                    speech_chunks = consecutive_speech
                    recording_started = time.monotonic()
                    continue

                frames.append(data)

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

            speech_duration = speech_chunks * chunk_seconds
            if speech_duration < self.min_speech_duration:
                logger.info(
                    "Speech was too short: %.2f seconds",
                    speech_duration,
                )
                return None

            native = self._decode(
                b"".join(frames),
                audio_format,
                channels,
            )

            if self.save_native_debug:
                native_path = output_path.with_name(
                    f"{output_path.stem}_native{output_path.suffix}"
                )
                with wave.open(str(native_path), "wb") as native_wav:
                    native_wav.setnchannels(channels)
                    native_wav.setsampwidth(2)
                    native_wav.setframerate(rate)
                    native_wav.writeframes(
                        self._to_int16(native).tobytes()
                    )
                logger.info(
                    "Saved native debug capture to %s",
                    native_path,
                )

            mono = self._select_channel(native)
            mono = self._resample(
                mono,
                rate,
                self.sample_rate,
            )
            pcm = self._to_int16(mono)

            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm.tobytes())

            self.last_capture_info = {
                "device_index": device["index"],
                "device_name": device["name"],
                "capture_rate": rate,
                "capture_channels": channels,
                "target_rate": self.sample_rate,
                "speech_duration": speech_duration,
            }

            logger.info(
                "Saved %s from %d Hz/%d ch as 16 kHz mono",
                output_path,
                rate,
                channels,
            )
            return output_path

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

    def _calibrate_threshold(self, stream: pyaudio.Stream, config: dict) -> float:
        """Sử dụng nửa giây audio môi trường khi không có ngưỡng cố định được đưa ra."""
        values = []

        for _ in range(
            max(1, round(self.calibration_duration * config["rate"] / self.chunk_size))
        ):
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
            raise ValueError(f"Định dạng audio không được hỗ trợ: {audio_format}")

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

    def _select_channel(self, samples: np.ndarray) -> np.ndarray:
        mode = self.channel_mode.strip().lower()
        if samples.shape[1] == 1:
            return samples[:, 0]
        if mode in {"auto", "best-energy", "beamformed"}:
            return self._best_channel(samples)
        if mode == "first":
            return samples[:, 0]
        if mode == "mix":
            return np.asarray(samples.mean(axis=1), dtype=np.float32)
        if mode.startswith("channel:"):
            channel_index = int(mode.split(":", 1)[1])
            if channel_index < 0 or channel_index >= samples.shape[1]:
                raise ValueError(
                    f"channel mode {self.channel_mode!r} is out of range "
                    f"for {samples.shape[1]} channel(s)"
                )
            return samples[:, channel_index]
        return self._best_channel(samples)

    @staticmethod
    def _best_channel(samples: np.ndarray) -> np.ndarray:
        """Giữ các kênh native cho đến khi chọn một kênh cho Whisper."""
        if samples.shape[1] == 1:
            return samples[:, 0]

        channel_rms = np.sqrt(
            np.mean(np.square(samples, dtype=np.float64), axis=0)
        )
        return samples[:, int(np.argmax(channel_rms))]

    @staticmethod
    def _resample(
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate == target_rate or samples.size == 0:
            return samples.astype(np.float32, copy=False)

        # ponytail: resampling tuyến tính là đủ cho STT; chỉ dùng resample_poly
        # nếu test chuyển cho thấy vấn đề chất lượng đo được.
        target_length = max(1, round(samples.size * target_rate / source_rate))
        return np.interp(
            np.linspace(0, samples.size - 1, target_length),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)

    @staticmethod
    def _to_int16(samples: np.ndarray) -> np.ndarray:
        return np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")

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


def _self_check() -> None:
    source = np.array([0.0, 1.0], dtype=np.float32)
    assert AudioRecorder._resample(source, 2, 4).shape == (4,)
    assert AudioRecorder._to_int16(source).dtype == np.dtype("<i2")


if __name__ == "__main__":
    _self_check()
    print("audio_recorder self-check passed")