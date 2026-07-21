from pathlib import Path
from typing import Optional
import importlib


class VietnameseSTT:
    """Vietnamese Speech-to-Text using Whisper."""
    
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize STT with Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
                        tiny is fastest for Raspberry Pi
        """
        try:
            whisper = importlib.import_module("whisper")
        except ImportError as exc:
            raise RuntimeError(
                "Whisper not installed. Install with: pip install openai-whisper"
            ) from exc
        
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
        result = self.model.transcribe(
            str(audio_path),
            language="vi",
            fp16=False,
        )
        return result["text"].strip()
    
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
