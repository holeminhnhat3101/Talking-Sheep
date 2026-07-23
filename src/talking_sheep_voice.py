"""Talking Sheep voice entry point.

Initializes all components once and runs a sequential conversation loop:

    microphone
    → native-format negotiation
    → optional multichannel/spatial processing
    → 16 kHz mono PhoWhisper input
    → LLM
    → Kokoro TTS + optional bleat
    → synchronous playback
    → repeat

Microphone failures use bounded retry delays so unplugged, unsupported, or
temporarily busy devices do not create a rapid traceback loop.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

# Ensure ``src`` imports work whether this file is executed directly or imported.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .config import (
        AUDIO_AUTO_CALIBRATE,
        AUDIO_CALIBRATION_DURATION,
        AUDIO_CAPTURE_CHANNELS,
        AUDIO_CAPTURE_RATE,
        AUDIO_CHANNEL_MODE,
        AUDIO_INPUT_DEVICE,
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_OUTPUT_DEVICE,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        CYCLE_RETRY_DELAY,
        DEFAULT_BLEATS_DIR,
        DEFAULT_DEVICE,
        DEFAULT_LOG_LEVEL,
        DEFAULT_RUNTIME_DIR,
        DEFAULT_VOICE,
        DEFAULT_INPUT_WAV,
        MAX_RECORDING_DURATION,
        MIC_RETRY_INITIAL_DELAY,
        MIC_RETRY_MAX_DELAY,
        MIN_SPEECH_DURATION,
        PRE_ROLL_DURATION,
        SILENCE_DURATION,
        SILENCE_THRESHOLD,
        STT_DEFAULT_MODEL,
    )
except ImportError:
    from src.config import (
        AUDIO_AUTO_CALIBRATE,
        AUDIO_CALIBRATION_DURATION,
        AUDIO_CAPTURE_CHANNELS,
        AUDIO_CAPTURE_RATE,
        AUDIO_CHANNEL_MODE,
        AUDIO_INPUT_DEVICE,
        AUDIO_MAX_WAIT_FOR_SPEECH,
        AUDIO_MINIMUM_AUTO_THRESHOLD,
        AUDIO_OUTPUT_DEVICE,
        AUDIO_SAVE_NATIVE_DEBUG,
        AUDIO_SPEECH_START_CHUNKS,
        AUDIO_THRESHOLD_MULTIPLIER,
        CYCLE_RETRY_DELAY,
        DEFAULT_BLEATS_DIR,
        DEFAULT_DEVICE,
        DEFAULT_LOG_LEVEL,
        DEFAULT_RUNTIME_DIR,
        DEFAULT_VOICE,
        DEFAULT_INPUT_WAV,
        MAX_RECORDING_DURATION,
        MIC_RETRY_INITIAL_DELAY,
        MIC_RETRY_MAX_DELAY,
        MIN_SPEECH_DURATION,
        PRE_ROLL_DURATION,
        SILENCE_DURATION,
        SILENCE_THRESHOLD,
        STT_DEFAULT_MODEL,
    )


logger = logging.getLogger("talking_sheep")

DeviceSelector = Optional[Union[int, str]]


def _parse_device_selector(value: str) -> DeviceSelector:
    """Accept either a PortAudio index or a partial/exact device name."""
    text = value.strip()
    if not text:
        return None

    try:
        index = int(text)
    except ValueError:
        return text

    if index < 0:
        raise argparse.ArgumentTypeError("device index must be non-negative")
    return index


def _parse_optional_positive_int(value: str) -> Optional[int]:
    text = value.strip().lower()
    if text in {"", "auto", "none", "unset"}:
        return None

    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive integer or 'auto'"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_optional_positive_float(value: str) -> Optional[float]:
    text = value.strip().lower()
    if text in {"", "auto", "none", "unset"}:
        return None

    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive number or 'auto'"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_silence_threshold(value: str) -> Optional[float]:
    """Use ``auto`` to enable per-device ambient-noise calibration."""
    text = value.strip().lower()
    if text in {"auto", "none", "unset"}:
        return None

    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "silence threshold must be non-negative or 'auto'"
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "silence threshold cannot be negative"
        )
    return parsed


def _parse_channel_mode(value: str) -> str:
    mode = value.strip().lower()
    valid = {"auto", "mix", "first", "best-energy", "beamformed"}

    if mode in valid:
        return mode

    if mode.startswith("channel:"):
        try:
            channel_index = int(mode.split(":", 1)[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "channel mode must be auto, mix, first, best-energy, "
                "beamformed, or channel:N"
            ) from exc

        if channel_index < 0:
            raise argparse.ArgumentTypeError(
                "channel:N requires a non-negative channel index"
            )
        return mode

    raise argparse.ArgumentTypeError(
        "channel mode must be auto, mix, first, best-energy, "
        "beamformed, or channel:N"
    )


def run_once(
    recorder: Any,
    stt: Any,
    llm: Any,
    tts: Any,
    player: Any,
    runtime_dir: Path,
    bleats_dir: Path,
    *,
    silence_threshold: Optional[float],
    silence_duration: float,
) -> bool:
    """Run one complete microphone-to-speaker interaction.

    Recording and playback are sequential and never overlap, preventing the
    microphone from recording the sheep's own response.

    Returns ``True`` only when a complete response was played.
    """
    try:
        from .voice_layer import create_spoken_response
    except ImportError:
        from src.voice_layer import create_spoken_response

    input_wav = recorder.capture_utterance(
        output_path=runtime_dir / DEFAULT_INPUT_WAV,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
    )
    if input_wav is None:
        logger.debug("No usable speech was captured.")
        return False


    transcript = stt.transcribe(input_wav).strip()
    if not transcript:
        logger.info("No usable Vietnamese transcription was produced.")
        return False

    logger.info("User transcript: %s", transcript)

    response = llm.generate_response(transcript).strip()
    if not response:
        logger.warning("LLM produced no usable response.")
        return False

    final_wav = create_spoken_response(
        response_text=response,
        tts=tts,
        bleats_dir=bleats_dir,
        runtime_dir=runtime_dir,
    )
    if final_wav is None:
        logger.warning("TTS produced no final audio file.")
        return False

    final_wav_path = Path(final_wav)
    if not final_wav_path.is_file():
        logger.warning("Final audio file does not exist: %s", final_wav_path)
        return False

    # Playback blocks until completion. Only then may the next recording begin.
    player.play_blocking(str(final_wav_path))
    return True


def run_conversation_loop(
    recorder: Any,
    stt: Any,
    llm: Any,
    tts: Any,
    player: Any,
    runtime_dir: Path,
    bleats_dir: Path,
    *,
    silence_threshold: Optional[float],
    silence_duration: float,
    mic_retry_initial_delay: float,
    mic_retry_max_delay: float,
    cycle_retry_delay: float,
) -> None:
    """Keep listening until interrupted without reloading persistent models."""
    try:
        from .audio_recorder import MicrophoneUnavailableError
    except ImportError:
        from src.audio_recorder import MicrophoneUnavailableError

    retry_delay = max(0.1, float(mic_retry_initial_delay))
    maximum_retry_delay = max(retry_delay, float(mic_retry_max_delay))
    ordinary_error_delay = max(0.1, float(cycle_retry_delay))
    last_microphone_error: Optional[str] = None

    try:
        while True:
            try:
                logger.info("Listening for the next utterance...")
                completed = run_once(
                    recorder,
                    stt,
                    llm,
                    tts,
                    player,
                    runtime_dir,
                    bleats_dir,
                    silence_threshold=silence_threshold,
                    silence_duration=silence_duration,
                )

                # Reaching this point means microphone discovery, negotiation,
                # stream opening, and capture all worked. Reset backoff even if
                # no usable speech was detected.
                retry_delay = max(0.1, float(mic_retry_initial_delay))
                last_microphone_error = None

                if completed:
                    logger.info("Conversation cycle complete.")

            except KeyboardInterrupt:
                raise

            except MicrophoneUnavailableError as exc:
                message = str(exc)

                if message != last_microphone_error:
                    logger.error("Microphone unavailable: %s", message)
                    last_microphone_error = message
                else:
                    logger.warning(
                        "Microphone is still unavailable; waiting before retry."
                    )

                logger.info(
                    "Re-enumerating microphone devices in %.1f seconds...",
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(maximum_retry_delay, retry_delay * 2.0)

            except Exception:
                # Unexpected cycle errors retain a traceback for diagnosis, but
                # a short delay prevents another persistent failure from
                # producing a millisecond-speed loop.
                logger.exception("Error in conversation cycle — recovering.")
                time.sleep(ordinary_error_delay)

    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
        logger.info("Exiting on Ctrl+C.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talking Sheep voice loop")

    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Kokoro voice name",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="TTS inference device",
    )
    parser.add_argument(
        "--stt-model",
        default=STT_DEFAULT_MODEL,
        help="PhoWhisper model size",
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help="LLM model root directory",
    )
    parser.add_argument(
        "--bleats-dir",
        default=DEFAULT_BLEATS_DIR,
        help="Path to bleat WAV files",
    )
    parser.add_argument(
        "--runtime-dir",
        default=DEFAULT_RUNTIME_DIR,
        help="Directory for temporary WAV files",
    )

    parser.add_argument(
        "--input-device",
        type=_parse_device_selector,
        default=None,
        metavar="INDEX_OR_NAME",
        help=(
            "Microphone PortAudio index or exact/partial device name. "
            "When omitted, use AUDIO_INPUT_DEVICE or automatic detection."
        ),
    )
    parser.add_argument(
        "--capture-rate",
        type=_parse_optional_positive_int,
        default=AUDIO_CAPTURE_RATE,
        metavar="HZ_OR_AUTO",
        help=(
            "Preferred native microphone rate. Use 'auto' to negotiate."
        ),
    )
    parser.add_argument(
        "--capture-channels",
        type=_parse_optional_positive_int,
        default=AUDIO_CAPTURE_CHANNELS,
        metavar="COUNT_OR_AUTO",
        help=(
            "Preferred native channel count. Use 'auto' to preserve and "
            "negotiate the microphone's supported channels."
        ),
    )
    parser.add_argument(
        "--channel-mode",
        type=_parse_channel_mode,
        default=AUDIO_CHANNEL_MODE,
        metavar="MODE",
        help=(
            "Channel processing: auto, mix, first, best-energy, "
            "beamformed, or channel:N."
        ),
    )
    parser.add_argument(
        "--silence-threshold",
        type=_parse_silence_threshold,
        default=SILENCE_THRESHOLD,
        metavar="VALUE_OR_AUTO",
        help=(
            "Fixed RMS speech threshold, or 'auto' for per-device "
            "ambient-noise calibration."
        ),
    )
    parser.add_argument(
        "--max-wait-for-speech",
        type=_parse_optional_positive_float,
        default=AUDIO_MAX_WAIT_FOR_SPEECH,
        metavar="SECONDS_OR_NONE",
        help=(
            "Maximum time to wait for speech before returning to the loop. "
            "Use 'none' for no timeout."
        ),
    )
    parser.add_argument(
        "--auto-calibrate",
        action=argparse.BooleanOptionalAction,
        default=bool(AUDIO_AUTO_CALIBRATE),
        help="Enable or disable ambient-noise calibration.",
    )
    parser.add_argument(
        "--save-native-audio",
        action=argparse.BooleanOptionalAction,
        default=bool(AUDIO_SAVE_NATIVE_DEBUG),
        help=(
            "Save the untouched native multichannel recording beside "
            "runtime/input.wav for debugging microphone arrays."
        ),
    )

    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List detected microphones and exit",
    )
    parser.add_argument(
        "--mic-retry-initial-delay",
        type=float,
        default=MIC_RETRY_INITIAL_DELAY,
        help="Initial microphone retry delay in seconds",
    )
    parser.add_argument(
        "--mic-retry-max-delay",
        type=float,
        default=MIC_RETRY_MAX_DELAY,
        help="Maximum microphone retry delay in seconds",
    )
    parser.add_argument(
        "--cycle-retry-delay",
        type=float,
        default=CYCLE_RETRY_DELAY,
        help="Delay after unexpected non-microphone cycle errors",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level",
    )

    args = parser.parse_args(argv)

    if args.mic_retry_initial_delay <= 0:
        parser.error("--mic-retry-initial-delay must be positive")
    if args.mic_retry_max_delay <= 0:
        parser.error("--mic-retry-max-delay must be positive")
    if args.cycle_retry_delay <= 0:
        parser.error("--cycle-retry-delay must be positive")

    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    bleats_dir = Path(args.bleats_dir)
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bleats_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing components...")

    try:
        from .audio_player import AudioPlayer
        from .audio_recorder import AudioRecorder
        from .chat_llm import LLMChat
        from .vietnamese_stt import VietnameseSTT
        from .voice_layer import KokoroTTS
    except ImportError:
        from src.audio_player import AudioPlayer
        from src.audio_recorder import AudioRecorder
        from src.chat_llm import LLMChat
        from src.vietnamese_stt import VietnameseSTT
        from src.voice_layer import KokoroTTS

    recorder: Optional[Any] = None
    player: Optional[Any] = None

    try:
        input_selector: DeviceSelector = (
            args.input_device
            if args.input_device is not None
            else AUDIO_INPUT_DEVICE
        )

        recorder = AudioRecorder(
            device_index=input_selector,
            pre_roll_duration=PRE_ROLL_DURATION,
            min_speech_duration=MIN_SPEECH_DURATION,
            max_recording_duration=MAX_RECORDING_DURATION,
            capture_rate=args.capture_rate,
            capture_channels=args.capture_channels,
            channel_mode=args.channel_mode,
            auto_calibrate=args.auto_calibrate,
            calibration_duration=AUDIO_CALIBRATION_DURATION,
            threshold_multiplier=AUDIO_THRESHOLD_MULTIPLIER,
            minimum_auto_threshold=AUDIO_MINIMUM_AUTO_THRESHOLD,
            speech_start_chunks=AUDIO_SPEECH_START_CHUNKS,
            max_wait_for_speech=args.max_wait_for_speech,
            save_native_debug=args.save_native_audio,
        )

        if args.list_mics:
            microphones = recorder.list_input_devices()
            if not microphones:
                print("No microphone input devices detected.")
                return

            print("Detected Microphone Input Devices:")
            for microphone in microphones:
                print(
                    f"  [{microphone['index']}] {microphone['name']} "
                    f"(Host API: {microphone['host_api']}, "
                    f"Channels: {microphone['channels']}, "
                    f"Default: {microphone['default_sample_rate']} Hz)"
                )
            return

        logger.info(
            "AudioRecorder ready "
            "(selector=%r, capture_rate=%s, capture_channels=%s, "
            "channel_mode=%s, threshold=%s).",
            input_selector,
            args.capture_rate or "auto",
            args.capture_channels or "auto",
            args.channel_mode,
            "auto" if args.silence_threshold is None else args.silence_threshold,
        )

        # Persistent PhoWhisper model.
        stt = VietnameseSTT(model_size=args.stt_model)
        logger.info("STT ready (model=%s).", args.stt_model)

        # Persistent LLM model.
        model_root = Path(args.model_root) if args.model_root else REPO_ROOT
        llm = LLMChat(model_root=model_root)
        logger.info("LLMChat ready.")

        # Persistent Kokoro engine.
        vendored_src = REPO_ROOT / "Kokoro-Vietnamese" / "src"
        if vendored_src.is_dir() and str(vendored_src) not in sys.path:
            sys.path.insert(0, str(vendored_src))

        kokoro_class = importlib.import_module(
            "kokoro_vietnamese"
        ).KokoroVietnamese
        tts_engine = kokoro_class(device=args.device, voice=args.voice)
        tts = KokoroTTS(tts_engine)
        logger.info(
            "Kokoro TTS ready (voice=%s, device=%s).",
            args.voice,
            args.device,
        )

        # Persistent audio player.
        player = AudioPlayer(device_index=AUDIO_OUTPUT_DEVICE)
        logger.info(
            "AudioPlayer ready (device=%s).",
            AUDIO_OUTPUT_DEVICE
            if AUDIO_OUTPUT_DEVICE is not None
            else "system default",
        )

        logger.info("All components initialized. Starting conversation loop.")
        print("\n🐑 Talking Sheep sẵn sàng! Nhấn Ctrl+C để thoát.\n")

        run_conversation_loop(
            recorder,
            stt,
            llm,
            tts,
            player,
            runtime_dir,
            bleats_dir,
            silence_threshold=args.silence_threshold,
            silence_duration=SILENCE_DURATION,
            mic_retry_initial_delay=args.mic_retry_initial_delay,
            mic_retry_max_delay=args.mic_retry_max_delay,
            cycle_retry_delay=args.cycle_retry_delay,
        )

    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                logger.debug("AudioRecorder cleanup failed", exc_info=True)

        if player is not None:
            try:
                player.close()
            except Exception:
                logger.debug("AudioPlayer cleanup failed", exc_info=True)


if __name__ == "__main__":
    main()