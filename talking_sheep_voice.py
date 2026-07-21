from pathlib import Path
from audio_recorder import AudioRecorder
from vietnamese_stt import VietnameseSTT
from vietnamese_tts import VietnameseTTS
from audio_player import AudioPlayer
from chat_phogpt_q8 import PhoGPTChat
from voice_layer import create_spoken_response


class TalkingSheepVoice:
    """Main voice layer for Talking Sheep application."""
    
    def __init__(
        self,
        model_root: Path = None,
        stt_model: str = "tiny",
        tts_model_dir: Path = None,
    ):
        """
        Initialize all voice components.
        
        Args:
            model_root: Path to LLM model directory
            stt_model: Whisper model size (tiny, base, small, medium, large)
            tts_model_dir: Path to sherpa-onnx TTS model directory
        """
        print("Initializing Talking Sheep Voice Layer...")
        
        self.recorder = AudioRecorder()
        self.stt = VietnameseSTT(model_size=stt_model)
        self.tts = VietnameseTTS(model_dir=tts_model_dir)
        self.player = AudioPlayer()
        self.llm = PhoGPTChat(model_root=model_root)
        
        # Create runtime directory
        self.runtime_dir = Path("runtime")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        
        print("Voice layer initialized.")
    
    def run_once(self) -> None:
        """Run one complete conversation cycle."""
        print("\n--- Starting conversation cycle ---")
        
        # Step 1: Record user input
        input_wav = self.recorder.capture_utterance()
        
        # Step 2: Transcribe to text
        print("Transcribing...")
        transcript = self.stt.transcribe(input_wav).strip()
        if not transcript:
            print("No speech detected.")
            return
        
        print(f"User said: {transcript}")
        
        # Step 3: Generate response with LLM
        print("Generating response...")
        response = self.llm.generate_response(transcript).strip()
        if not response:
            print("LLM returned empty response.")
            return
        
        print(f"Sheep responds: {response}")
        
        # Step 4: Create spoken response with TTS and bleats
        print("Synthesizing speech...")
        final_wav = create_spoken_response(
            response_text=response,
            tts=self.tts,
            runtime_dir=self.runtime_dir,
        )
        
        # Step 5: Play final audio
        print("Playing response...")
        self.player.play_blocking(str(final_wav))
        
        print("--- Conversation cycle complete ---\n")
    
    def run_continuous(self) -> None:
        """Run continuous conversation loop."""
        print("Talking Sheep Voice Layer")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                self.run_once()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.recorder.__del__()
            self.player.__del__()


def main():
    """Main entry point."""
    voice = TalkingSheepVoice()
    voice.run_continuous()


if __name__ == "__main__":
    main()
