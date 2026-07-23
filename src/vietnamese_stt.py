from pathlib import Path
from typing import Optional
import importlib
import os

try:
    from .config import DEFAULT_STT_MODEL, WHISPER_ALLOWED_MODELS
except ImportError:
    from src.config import DEFAULT_STT_MODEL, WHISPER_ALLOWED_MODELS


class VietnameseSTT:
    """Vietnamese Speech-to-Text using Whisper."""
    
    def __init__(self, model_size: str | None = None):
        """
        Initialize STT with Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
                        tiny is fastest for Raspberry Pi
        """
        model_size = model_size or os.environ.get("WHISPER_MODEL", DEFAULT_STT_MODEL)
        if model_size not in WHISPER_ALLOWED_MODELS:
            raise ValueError(
                f"WHISPER_MODEL must be one of: {', '.join(WHISPER_ALLOWED_MODELS)}"
            )
        try:
            whisper = importlib.import_module("whisper")
        except ImportError as exc:
            raise RuntimeError(
                "Whisper not installed. Install with: pip install openai-whisper"
            ) from exc
        
        try:
            from importlib.metadata import version
            assert hasattr(whisper, "load_model")
            version("openai-whisper")
        except Exception as exc:
            raise RuntimeError("The installed Whisper package is not openai-whisper.") from exc

        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe audio file to Vietnamese text.
        
        Args:
            audio_path: Path to WAV file
            
        Returns:
            Transcribed text in Vietnamese
        """
        try:
            result = self.model.transcribe(str(audio_path), language="vi", task="transcribe", fp16=False)
            return str(result.get("text", "")).strip()
        except Exception:
            return ""
    
    def transcribe_with_timestamps(self, audio_path: Path) -> dict:
        """
        Transcribe with segment timestamps.
        
        Args:
            audio_path: Path to WAV file
            
        Returns:
            Dict with text and segments
        """
        result = self.model.transcribe(
            str(audio_path),
            language="vi",
            fp16=False,
        )
        return result
