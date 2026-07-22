"""Talking Sheep voice entry point.

Initializes all components once and runs a sequential conversation loop:
  microphone → STT → PhoGPT → TTS + bleat → speaker → repeat.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("talking_sheep")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Talking Sheep voice loop")
    p.add_argument("--voice", default="diem_trinh", help="Kokoro voice name")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="TTS device")
    p.add_argument("--stt-model", default="tiny", help="Whisper model size")
    p.add_argument("--model-root", default=None, help="PhoGPT model root directory")
    p.add_argument("--bleats-dir", default="assets/bleats", help="Path to bleat WAV files")
    p.add_argument("--runtime-dir", default="runtime", help="Directory for temporary WAVs")
    p.add_argument("--log-level", default="INFO", help="Logging level")
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
        from .chat_phogpt_q8 import PhoGPTChat
        from .audio_player import AudioPlayer
        from .voice_layer import create_spoken_response
    except ImportError:
        from src.audio_recorder import AudioRecorder
        from src.vietnamese_stt import VietnameseSTT
        from src.chat_phogpt_q8 import PhoGPTChat
        from src.audio_player import AudioPlayer
        from src.voice_layer import create_spoken_response

    recorder = AudioRecorder()
    logger.info("AudioRecorder ready.")

    # Speech-to-Text
    stt = VietnameseSTT(model_size=args.stt_model)
    logger.info("STT ready (model=%s).", args.stt_model)

    # LLM
    model_root = Path(args.model_root) if args.model_root else repo_root
    llm = PhoGPTChat(model_root=model_root)
    logger.info("PhoGPTChat ready.")

    # TTS (Kokoro Vietnamese — loaded once)
    try:
        from kokoro_vietnamese import KokoroVietnamese
    except ImportError:
        vendored_src = repo_root / "Kokoro-Vietnamese" / "src"
        if vendored_src.is_dir() and str(vendored_src) not in sys.path:
            sys.path.insert(0, str(vendored_src))
        from kokoro_vietnamese import KokoroVietnamese

    tts = KokoroVietnamese(device=args.device, voice=args.voice)
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
        while True:
            try:
                # 1. Record
                input_wav = recorder.capture_utterance(
                    output_path=runtime_dir / "input.wav",
                )

                # 2. Transcribe
                logger.info("Transcribing...")
                transcript = stt.transcribe(input_wav).strip()
                if not transcript:
                    logger.info("Empty transcript — returning to listening.")
                    continue

                logger.info("User: %s", transcript)

                # 3. Generate response
                logger.info("Generating LLM response...")
                response = llm.generate_response(transcript).strip()
                if not response:
                    logger.warning("LLM returned empty response — returning to listening.")
                    continue

                logger.info("Sheep: %s", response)

                # 4. Synthesize + compose with optional bleat
                logger.info("Synthesizing speech...")
                final_wav = create_spoken_response(
                    response_text=response,
                    tts=tts,
                    bleats_dir=bleats_dir,
                    runtime_dir=runtime_dir,
                )

                # 5. Play (recording is implicitly disabled — sequential flow)
                logger.info("Playing response...")
                player.play_blocking(str(final_wav))

                logger.info("Cycle complete.")

            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Error in conversation cycle — recovering.")

    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
        logger.info("Exiting on Ctrl+C.")
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
