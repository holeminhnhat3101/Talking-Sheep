from pathlib import Path
from typing import Optional
import importlib
import numpy as np
import wave


class VietnameseTTS:
    """Vietnamese Text-to-Speech using sherpa-onnx."""
    
    def __init__(
        self,
        model_dir: Path = None,
        voice: str = "vits-vietnamese",
    ):
        """
        Initialize TTS with sherpa-onnx.
        
        Args:
            model_dir: Path to sherpa-onnx model directory
            voice: Voice model identifier
        """
        try:
            self.sherpa_onnx = importlib.import_module("sherpa_onnx")
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx not installed. Install with: pip install sherpa-onnx"
            ) from exc
        
        if model_dir is None:
            model_dir = Path("models/tts")
        
        self.model_dir = model_dir
        self.voice = voice
        
        # Initialize sherpa-onnx TTS
        # Note: You'll need to download Vietnamese VITS models
        # from https://github.com/k2-fsa/sherpa-onnx/releases
        self._init_model()
        
    def _init_model(self):
        """Initialize the sherpa-onnx TTS model."""
        # This is a placeholder - actual model paths depend on downloaded models
        # Typical Vietnamese VITS model structure:
        # - model.onnx (the main model)
        # - tokens.txt (vocabulary)
        # - config file (if needed)
        
        model_path = self.model_dir / "model.onnx"
        tokens_path = self.model_dir / "tokens.txt"
        
        if not model_path.exists() or not tokens_path.exists():
            raise RuntimeError(
                f"Vietnamese TTS models not found in {self.model_dir}. "
                "Download Vietnamese VITS models from sherpa-onnx releases."
            )
        
        # Create offline TTS config
        config = self.sherpa_onnx.OfflineTtsConfig(
            model=self.sherpa_onnx.OfflineTtsModelConfig(
                vits=self.sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    num_threads=2,
                ),
                provider="cpu",  # Use CPU for Raspberry Pi
                debug=False,
            ),
            rule_fsts="",  # Optional: path to rule FST for text normalization
            max_num_sentences=10,  # Maximum sentences per batch
        )
        
        self.tts = self.sherpa_onnx.OfflineTts(config)
        
    def synthesize(self, text: str, output_path: str) -> None:
        """
        Synthesize text to WAV file.
        
        Args:
            text: Vietnamese text to synthesize
            output_path: Path for output WAV file
        """
        # Generate audio
        audio = self.tts.generate(text)
        
        # Save to WAV file
        self._save_wav(audio.samples, audio.sample_rate, output_path)
    
    def _save_wav(self, samples: np.ndarray, sample_rate: int, output_path: str) -> None:
        """Save numpy audio array to WAV file."""
        # Ensure samples are in correct format (int16)
        if samples.dtype != np.int16:
            # Convert float32 to int16
            if samples.dtype == np.float32 or samples.dtype == np.float64:
                samples = (samples * 32767).astype(np.int16)
            else:
                samples = samples.astype(np.int16)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
