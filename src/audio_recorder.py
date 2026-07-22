import pyaudio
import wave
from pathlib import Path
from typing import Optional


class AudioRecorder:
    """Record audio from microphone with VAD."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        
    def capture_utterance(
        self,
        output_path: Optional[Path] = None,
        silence_threshold: int = 500,
        silence_duration: float = 1.0,
    ) -> Path:
        """Record a single utterance with VAD-based stopping."""
        if output_path is None:
            output_path = Path("runtime/input.wav")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        
        frames = []
        silence_frames = 0
        max_silence_frames = int(silence_duration * self.sample_rate / self.chunk_size)
        
        print("Recording... (speak now)")
        
        try:
            while True:
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # Simple VAD: check RMS energy
                audio_data = bytearray(data)
                rms = self._calculate_rms(audio_data)
                
                if rms < silence_threshold:
                    silence_frames += 1
                    if silence_frames > max_silence_frames:
                        break
                else:
                    silence_frames = 0
                    
        finally:
            stream.stop_stream()
            stream.close()
        
        # Save to WAV file
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"Saved to {output_path}")
        return output_path
    
    def _calculate_rms(self, data: bytearray) -> float:
        """Calculate RMS energy of audio data."""
        import array
        samples = array.array('h', data)
        sum_squares = sum(s * s for s in samples)
        return (sum_squares / len(samples)) ** 0.5 if samples else 0
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.audio.terminate()
