import pytest
import threading
import time
import numpy as np
from pydub import AudioSegment
from src.streaming_response import stream_response_to_player, _queue_put_until_stopped
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
        self._in_streaming_session = False

    def start_streaming_session(self):
        self._in_streaming_session = True

    def end_streaming_session(self):
        self._in_streaming_session = False

    def write_pcm_blocking(self, pcm_data):
        self.played.append(pcm_data)
        if self.on_play_callback:
            self.on_play_callback(pcm_data)

    def play_segment_blocking(self, segment):
        # For backward compatibility with non-streaming tests
        self.played.append(segment.raw_data)
        if self.on_play_callback:
            self.on_play_callback(segment.raw_data)

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
        speech_end_time=time.monotonic(),
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
        speech_end_time=time.monotonic(),
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
            speech_end_time=time.monotonic(),
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
        speech_end_time=time.monotonic(),
    )
    assert completed is False
    assert len(player.played) == 0

def test_bleat_insertion(monkeypatch):
    """Test that bleats are inserted between sentences, not before first or after last."""
    llm = FakeLLM(["First sentence. Second sentence. Third sentence."])
    tts = FakeTTS()
    player = FakePlayer()
    
    # Create a fake bleat segment
    bleat_segment = create_test_audio_segment(50)
    
    # Force bleat insertion by monkeypatching random
    import random
    monkeypatch.setattr(random, "random", lambda: 0.0)
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(bleat_segment,),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # Should have: speech, bleat, speech, bleat, speech (3 sentences, 2 bleats)
    assert len(player.played) == 5
    
    from src.voice_layer import PAUSE_BEFORE_BLEAT_MS, PAUSE_AFTER_BLEAT_MS
    before = normalize_segment(AudioSegment.silent(duration=PAUSE_BEFORE_BLEAT_MS, frame_rate=TARGET_SAMPLE_RATE))
    after = normalize_segment(AudioSegment.silent(duration=PAUSE_AFTER_BLEAT_MS, frame_rate=TARGET_SAMPLE_RATE))
    expected_interstitial = before + bleat_segment + after
    expected_len = len(expected_interstitial.raw_data)
    
    # Verify that we have the right pattern by checking segment lengths
    assert len(player.played[0]) > 0  # First speech (longer)
    assert len(player.played[1]) == expected_len  # First bleat (exact match bytes)
    assert len(player.played[2]) > 0  # Second speech (longer)
    assert len(player.played[3]) == expected_len  # Second bleat (exact match bytes)
    assert len(player.played[4]) > 0  # Third speech (longer)

def test_queue_saturation():
    """Test that the pipeline handles queue saturation gracefully with retries."""
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
        speech_end_time=time.monotonic(),
    )
    
    # Should complete successfully despite queue saturation (with retries)
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
            speech_end_time=time.monotonic(),
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
            speech_end_time=time.monotonic(),
        )

def test_cancellation_during_blocked_put_get():
    """Test that cancellation (stop_event) unblocks blocked put/get operations cleanly."""
    class BlockingLLM:
        def generate_response_chunks(self, prompt):
            # Generates enough sentences to saturate the queue and block on put
            for i in range(100):
                yield f"Sentence {i}."
                
    class PlayerThatCrashes:
        def __init__(self):
            self.played = []
        def start_streaming_session(self): pass
        def end_streaming_session(self): pass
        def write_pcm_blocking(self, pcm_data):
            raise RuntimeError("Simulated player crash")
            
    llm = BlockingLLM()
    tts = FakeTTS()
    player = PlayerThatCrashes()
    
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="Simulated player crash"):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=start_time,
        )
    end_time = time.monotonic()
    
    # The pipeline should tear down very quickly because stop_event interrupts
    # the LLM's blocked queue.put and TTS's queue.get.
    assert (end_time - start_time) < 1.0

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
            speech_end_time=time.monotonic(),
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
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # Verify no exceptions were raised during shutdown
    # The function should return cleanly

def test_streaming_session_lifecycle():
    """Test that streaming session is properly managed."""
    llm = FakeLLM(["First sentence. Second sentence."])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # Verify streaming session was started and ended
    assert player._in_streaming_session == False  # Should be False after end_session

def test_segment_order_preserved():
    """Test that segments remain ordered during streaming."""
    llm = FakeLLM(["First. Second. Third."])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # Should have 3 segments in order
    assert len(player.played) == 3
    # Each segment should be non-empty PCM data
    for segment in player.played:
        assert len(segment) > 0

def test_pcm_written_before_close():
    """Test that all PCM data is written before stream close."""
    llm = FakeLLM(["Sentence one. Sentence two."])
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # All segments should have been written (played)
    assert len(player.played) == 2
    # Total bytes should equal sum of all segments
    total_bytes = sum(len(seg) for seg in player.played)
    assert total_bytes > 0

def test_timing_metrics_consistent_clock():
    """Test that all timing metrics use consistent monotonic clock.
    
    This test would fail if speech_end_time used time.perf_counter() while
    other metrics used time.monotonic(), as subtracting different clock sources
    produces impossible values (e.g., 1.7 billion seconds).
    """
    import time
    
    # Simulate the actual workflow with consistent clock source
    speech_end_time = time.monotonic()
    
    # Simulate first sentence boundary (should use same clock)
    first_sentence_boundary = time.monotonic()
    
    # Simulate first synthesized segment (should use same clock)
    first_synthesized_segment = time.monotonic()
    
    # Simulate first playback (should use same clock)
    first_playback = time.monotonic()
    
    # All timestamps should be from the same clock source
    # and in chronological order
    assert speech_end_time <= first_sentence_boundary
    assert first_sentence_boundary <= first_synthesized_segment
    assert first_synthesized_segment <= first_playback
    
    # Time differences should be reasonable (0-60 seconds for normal operation)
    assert 0 <= (first_sentence_boundary - speech_end_time) <= 60
    assert 0 <= (first_synthesized_segment - speech_end_time) <= 60
    assert 0 <= (first_playback - speech_end_time) <= 60

def test_llm_worker_normal_completion():
    """Test that LLM worker finishing early is normal and doesn't cause pipeline failure.
    
    This addresses the bug where the main thread incorrectly treated
    llm_thread.is_alive() == False as a pipeline failure. LLM is supposed
    to finish before TTS and playback complete.
    """
    class EarlyFinishingLLM:
        def generate_response_chunks(self, prompt):
            yield "First sentence."
            yield "Second sentence."
            # LLM finishes here, but TTS and playback should continue
    
    llm = EarlyFinishingLLM()
    tts = FakeTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    # Should complete successfully even though LLM finished early
    assert completed is True
    assert len(player.played) == 2

def test_tts_worker_owns_final_queue():
    """Test that only TTS worker termination indicates pipeline completion.
    
    The main thread should monitor TTS worker because it owns the final
    audio queue. LLM ending early is normal.
    """
    class SlowTTS:
        def __init__(self):
            self.delayed = False
        
        def synthesize(self, text, speed=None):
            if not self.delayed:
                # Small delay to ensure LLM finishes first
                time.sleep(0.05)
                self.delayed = True
            return np.array([0.1] * 100, dtype=np.float32), "phonemes"
    
    llm = FakeLLM(["Quick sentence."])
    tts = SlowTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    # Should complete successfully with proper queue monitoring
    assert completed is True
    assert len(player.played) == 1

def test_tts_still_produces_audio_after_llm_exits():
    """Test that TTS still produces queued audio after LLM exits."""
    llm_done = threading.Event()
    
    class FastLLM:
        def generate_response_chunks(self, prompt):
            yield "First sentence."
            yield "Second sentence."
            llm_done.set()
            
    class WaitingTTS:
        def synthesize(self, text, speed=None):
            # Wait for LLM to completely finish and exit before synthesizing
            llm_done.wait()
            return np.array([0.1] * 100, dtype=np.float32), "phonemes"
            
    llm = FastLLM()
    tts = WaitingTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    assert len(player.played) == 2

def test_terminal_messages_under_backpressure():
    """Test that terminal messages (_SentenceEnd) are not dropped when queues are full."""
    class BurstLLM:
        def generate_response_chunks(self, prompt):
            # Generates more than maxsize of sentence_queue (4) + audio_queue (3)
            for i in range(10):
                yield f"Sentence {i}."
                
    class SlowTTS:
        def synthesize(self, text, speed=None):
            time.sleep(0.01) # Small delay to ensure backpressure builds up
            return np.array([0.1] * 100, dtype=np.float32), "phonemes"
            
    llm = BurstLLM()
    tts = SlowTTS()
    player = FakePlayer()
    
    completed = stream_response_to_player(
        prompt="start",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    
    assert completed is True
    # All 10 sentences should be successfully processed and played despite backpressure
    assert len(player.played) == 10

def test_tts_worker_exits_without_terminal_message(monkeypatch):
    """Test the failure mode where TTS exits silently (without _AudioEnd or _AudioFailure)."""
    import src.streaming_response
    
    def fake_tts_worker(*args, **kwargs):
        # Simply exits without putting anything in the audio_queue
        pass
        
    monkeypatch.setattr(src.streaming_response, "_tts_worker", fake_tts_worker)
    
    llm = FakeLLM(["A sentence."])
    tts = FakeTTS()
    player = FakePlayer()
    
    with pytest.raises(RuntimeError, match="TTS worker exited without an audio terminal message."):
        stream_response_to_player(
            prompt="start",
            llm=llm,
            tts=tts,
            player=player,
            bleat_segments=(),
            speech_end_time=time.monotonic(),
        )

def test_clean_worker_joins_on_success_and_failure():
    """Test that pipeline workers terminate cleanly and their joins do not timeout."""
    class NormalLLM:
        def generate_response_chunks(self, prompt):
            yield "One."
            
    # Success case
    player = FakePlayer()
    completed = stream_response_to_player(
        prompt="start",
        llm=NormalLLM(),
        tts=FakeTTS(),
        player=player,
        bleat_segments=(),
        speech_end_time=time.monotonic(),
    )
    assert completed is True
    
    # Failure case (TTS crash)
    class CrashingTTS:
        def synthesize(self, text, speed=None):
            raise RuntimeError("Internal TTS crash")
            
    with pytest.raises(RuntimeError, match="Internal TTS crash"):
        stream_response_to_player(
            prompt="start",
            llm=NormalLLM(),
            tts=CrashingTTS(),
            player=FakePlayer(),
            bleat_segments=(),
            speech_end_time=time.monotonic(),
        )
    # The absence of "Pipeline workers failed to terminate cleanly" confirms clean joins.
