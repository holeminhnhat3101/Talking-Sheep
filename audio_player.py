import pyaudio
import wave
from pathlib import Path


class AudioPlayer:
    """Play WAV files through speaker."""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        
    def play_blocking(self, audio_path: str) -> None:
        """
        Play audio file and block until completion.
        
        Args:
            audio_path: Path to WAV file
        """
        with wave.open(audio_path, 'rb') as wf:
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            stream.stop_stream()
            stream.close()
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.audio.terminate()
