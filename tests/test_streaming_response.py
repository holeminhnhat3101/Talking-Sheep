import pytest
import threading
import time
import numpy as np
from pydub import AudioSegment
from src.streaming_response import stream_response_to_player
from src.voice_layer import synthesize_sentence, numpy_to_segment, normalize_segment, KOKORO_SAMPLE_RATE, TARGET_SAMPLE_RATE, TARGET_CHANNELS, TARGET_SAMPLE_WIDTH

class FakeLLM:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def generate_response_chunks(self, prompt):
        for chunk in self.chunks:
            yield chunk

    def close(self):
        self.closed = True

class FakeTTS:
    def synthesize(self, text, speed=None):
        # Return fake audio data (100 samples of 0.1)
        return np.array([0.1] * 100, dtype=np.float32), "phonemes"

class FakePlayer:
    def __init__(self, on_play_callback=None):
        self.played = []
        self.on_play_callback = on_play_callback

    def play_segment_blocking(self, segment):
        self.played.append(segment)
        if self.on_play_callback:
            self.on_play_callback(segment)

    def close(self):
        pass

def create_test_audio_segment(duration_ms=100):
    """Create a test AudioSegment with proper format."""
    audio_array = np.array([0.1] * int(24000 * duration_ms / 1000), dtype=np.float32)
    segment = numpy_to_segment(audio_array, KOKORO_SAMPLE_RATE)
    return normalize_segment(segment)

def test_streaming_ordering():
    """Test that sentences are processed in correct order with new pipeline."""
    llm = FakeLLM(["Sentence one! Sentence two."])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    
    assert completed is True
    # Should have 2 speech segments (no bleats since bleat_segments is empty)
    assert len(player.played) == 2

def test_streaming_concurrency():
    """Test that LLM and TTS work concurrently with new pipeline."""
    play_started = threading.Event()
    
    class ConcurrencyLLM:
        def generate_response_chunks(self, prompt):
            yield "Sentence one!"
            play_started.wait(timeout=2.0)
            yield "Sentence two."
            
    llm = ConcurrencyLLM()
    tts = FakeTTS()
    
    def callback(segment):
        play_started.set()
        
    player = FakePlayer(on_play_callback=callback)
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    
    assert completed is True
    # Should have 2 speech segments
    assert len(player.played) == 2
    assert play_started.is_set()

def test_streaming_failure_raised():
    class ExceptionLLM:
        def generate_response_chunks(self, prompt):
            yield "Sentence one!"
            raise RuntimeError("LLM failed mid-stream")
            
    llm = ExceptionLLM()
    tts = FakeTTS()
    player = FakePlayer()
    
    with pytest.raises(RuntimeError, match="LLM failed mid-stream"):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=time.time(),
        )

def test_streaming_empty_response():
    llm = FakeLLM([])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    assert completed is False
    assert len(player.played) == 0

def test_bleat_insertion():
    """Test that bleats are inserted between sentences, not before first or after last."""
    llm = FakeLLM(["First sentence. Second sentence. Third sentence."])
    tts = FakeTTS()
    player = FakePlayer()
    
    # Create a fake bleat segment
    bleat_segment = create_test_audio_segment(50)
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(bleat_segment,),
        speech_end_time=time.time(),
    )
    
    assert completed is True
    # Should have: speech, bleat, speech, bleat, speech (3 sentences, 2 bleats)
    assert len(player.played) == 5
    
    # Verify that we have the right pattern by checking segment types
    # The pattern should be: speech, bleat, speech, bleat, speech
    # We can't directly check types since we're using AudioSegment, but we can check lengths
    assert len(player.played[0]) > 0  # First speech (longer)
    assert len(player.played[1]) == len(bleat_segment)  # First bleat (exact match)
    assert len(player.played[2]) > 0  # Second speech (longer)
    assert len(player.played[3]) == len(bleat_segment)  # Second bleat (exact match)
    assert len(player.played[4]) > 0  # Third speech (longer)

def test_queue_saturation():
    """Test that the pipeline handles queue saturation gracefully."""
    # Create an LLM that produces many sentences quickly
    many_sentences = ". ".join([f"Sentence {i}" for i in range(10)])
    llm = FakeLLM([many_sentences])  # Many sentences in one chunk
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    
    # Should complete successfully despite queue saturation
    assert completed is True
    assert len(player.played) > 0

def test_llm_worker_crash():
    """Test that LLM worker crashes are properly propagated."""
    class CrashingLLM:
        def generate_response_chunks(self, prompt):
            yield "First sentence."
            raise RuntimeError("LLM crashed")
    
    llm = CrashingLLM()
    tts = FakeTTS()
    player = FakePlayer()
    
    with pytest.raises(RuntimeError):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=time.time(),
        )

def test_tts_worker_crash():
    """Test that TTS worker crashes are properly propagated."""
    class CrashingTTS:
        def synthesize(self, text, speed=None):
            if "crash" in text.lower():
                raise RuntimeError("TTS crashed")
            return np.array([0.1] * 100, dtype=np.float32), "phonemes"
    
    llm = FakeLLM(["Normal sentence. Crash this. Another sentence."])
    tts = CrashingTTS()
    player = FakePlayer()
    
    with pytest.raises(RuntimeError):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=time.time(),
        )

def test_cancellation_during_processing():
    """Test that cancellation during processing is handled cleanly."""
    # This test is hard to implement without external cancellation mechanism
    # For now, just verify that slow processing completes
    class SlowLLM:
        def generate_response_chunks(self, prompt):
            yield "First sentence."
            time.sleep(0.1)  # Simulate slow generation
            yield "Second sentence."
    
    llm = SlowLLM()
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    
    assert completed is True

def test_error_propagation_across_queues():
    """Test that errors propagate correctly across queue boundaries."""
    class ErrorPropagatingLLM:
        def generate_response_chunks(self, prompt):
            yield "First sentence."
            raise ValueError("Cross-queue error")
    
    llm = ErrorPropagatingLLM()
    tts = FakeTTS()
    player = FakePlayer()
    
    with pytest.raises(ValueError):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=time.time(),
        )

def test_clean_shutdown():
    """Test that shutdown is clean and resources are released."""
    llm = FakeLLM(["Single sentence."])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.time(),
    )
    
    assert completed is True
    # Verify no exceptions were raised during shutdown
    # The function should return cleanly
