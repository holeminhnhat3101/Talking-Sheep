from pathlib import Path
import os

try:
    from .config import (
        STT_DEFAULT_MODEL,
        STT_MODEL_IDS,
        STT_ALLOWED_MODELS,
    )
except ImportError:
    from src.config import (
        STT_DEFAULT_MODEL,
        STT_MODEL_IDS,
        STT_ALLOWED_MODELS,
    )


class VietnameseSTT:
    """Vietnamese speech-to-text using PhoWhisper."""

    def __init__(self, model_size: str | None = None):
        model_size = model_size or os.getenv(
            "PHOWHISPER_MODEL",
            STT_DEFAULT_MODEL,
        )

        if model_size not in STT_ALLOWED_MODELS:
            raise ValueError(
                f"PHOWHISPER_MODEL must be one of: "
                f"{', '.join(STT_ALLOWED_MODELS)}"
            )

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "PhoWhisper dependencies are missing. "
                "Install requirements-rpi.txt."
            ) from exc

        model_id = STT_MODEL_IDS[model_size]
        print(f"Loading PhoWhisper model: {model_id}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=-1,
        )

    def transcribe(self, audio_path: Path) -> str:
        try:
            result = self.pipe(
                str(audio_path),
                generate_kwargs={
                    "language": "vi",
                    "task": "transcribe",
                },
            )
            return str(result.get("text", "")).strip()
        except Exception:
            return ""

    def transcribe_with_timestamps(self, audio_path: Path) -> dict:
        return dict(
            self.pipe(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs={
                    "language": "vi",
                    "task": "transcribe",
                },
            )
        )