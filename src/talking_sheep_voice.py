"""Talking Sheep voice entry point.

Initializes all components once and runs a sequential conversation loop:
  microphone → STT → PhoGPT → TTS + bleat → speaker → repeat.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

try:
    from .config import (
        DEFAULT_BLEATS_DIR,
        DEFAULT_DEVICE,
        DEFAULT_LOG_LEVEL,
        DEFAULT_RUNTIME_DIR,
        DEFAULT_STT_MODEL,
        DEFAULT_VOICE,
    )
except ImportError:
    from src.config import (
        DEFAULT_BLEATS_DIR,
        DEFAULT_DEVICE,
        DEFAULT_LOG_LEVEL,
        DEFAULT_RUNTIME_DIR,
        DEFAULT_STT_MODEL,
        DEFAULT_VOICE,
    )

logger = logging.getLogger("talking_sheep")


def run_once(recorder, stt, llm, tts, player, runtime_dir: Path, bleats_dir: Path) -> None:
    """Run one complete microphone-to-speaker interaction.

    Recording is performed before playback and never concurrently with it,
    which prevents microphone feedback from the sheep's response.
    """
    try:
        from .voice_layer import create_spoken_response
    except ImportError:
        from src.voice_layer import create_spoken_response

    input_wav = recorder.capture_utterance(output_path=runtime_dir / "input.wav")
    transcript = stt.transcribe(input_wav).strip()
    if not transcript:
        return

    response = llm.generate_response(transcript).strip()
    if not response:
        return

    final_wav = create_spoken_response(
        response_text=response,
        tts=tts,
        bleats_dir=bleats_dir,
        runtime_dir=runtime_dir,
    )
    player.play_blocking(str(final_wav))


def run_conversation_loop(recorder, stt, llm, tts, player, runtime_dir: Path, bleats_dir: Path) -> None:
    """Keep listening until interrupted, recovering from one-cycle errors."""
    try:
        while True:
            try:
                logger.info("Listening for the next utterance...")
                run_once(recorder, stt, llm, tts, player, runtime_dir, bleats_dir)
                logger.info("Cycle complete.")
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Error in conversation cycle — recovering.")
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
        logger.info("Exiting on Ctrl+C.")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Talking Sheep voice loop")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Kokoro voice name")
    p.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"], help="TTS device")
    p.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="Whisper model size")
    p.add_argument("--model-root", default=None, help="PhoGPT model root directory")
    p.add_argument("--bleats-dir", default=DEFAULT_BLEATS_DIR, help="Path to bleat WAV files")
    p.add_argument("--runtime-dir", default=DEFAULT_RUNTIME_DIR, help="Directory for temporary WAVs")
    p.add_argument("--input-device", type=int, default=None, help="Microphone input device index for PortAudio/ALSA")
    p.add_argument("--list-mics", action="store_true", help="List detected microphones and exit")
    p.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    bleats_dir = Path(args.bleats_dir)
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize components (once, at startup)
    # ------------------------------------------------------------------
    logger.info("Initializing components...")

    # Project root setup for imports
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Audio recorder
    try:
        from .audio_recorder import AudioRecorder
        from .vietnamese_stt import VietnameseSTT
        from .chat_phogpt import PhoGPTChat
        from .audio_player import AudioPlayer
        from .voice_layer import KokoroTTS
    except ImportError:
        from src.audio_recorder import AudioRecorder
        from src.vietnamese_stt import VietnameseSTT
        from src.chat_phogpt import PhoGPTChat
        from src.audio_player import AudioPlayer
        from src.voice_layer import KokoroTTS

    recorder = AudioRecorder(device_index=args.input_device)

    if args.list_mics:
        print("Detected Microphone Input Devices:")
        for mic in recorder.list_input_devices():
            print(f"  [{mic['index']}] {mic['name']} (Channels: {mic['channels']}, SampleRate: {mic['default_sample_rate']}Hz)")
        return

    logger.info("AudioRecorder ready (device_index=%s).", recorder.device_index)

    # Speech-to-Text
    stt = VietnameseSTT(model_size=args.stt_model)
    logger.info("STT ready (model=%s).", args.stt_model)

    # LLM
    model_root = Path(args.model_root) if args.model_root else repo_root
    llm = PhoGPTChat(model_root=model_root)
    logger.info("PhoGPTChat ready.")

    # TTS (Kokoro Vietnamese — loaded once)
    vendored_src = repo_root / "Kokoro-Vietnamese" / "src"
    if vendored_src.is_dir() and str(vendored_src) not in sys.path:
        sys.path.insert(0, str(vendored_src))
    kokoro_class = importlib.import_module("kokoro_vietnamese").KokoroVietnamese

    tts_engine = kokoro_class(device=args.device, voice=args.voice)
    tts = KokoroTTS(tts_engine)
    logger.info("Kokoro TTS ready (voice=%s, device=%s).", args.voice, args.device)

    # Audio player
    player = AudioPlayer()
    logger.info("AudioPlayer ready.")

    logger.info("All components initialized.  Starting conversation loop.")
    print("\n🐑 Talking Sheep sẵn sàng!  Nhấn Ctrl+C để thoát.\n")

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------
    try:
        run_conversation_loop(recorder, stt, llm, tts, player, runtime_dir, bleats_dir)
    finally:
        try:
            recorder.__del__()
        except Exception:
            pass
        try:
            player.__del__()
        except Exception:
            pass


if __name__ == "__main__":
    main()
