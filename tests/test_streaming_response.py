import pytest
import threading
import time
from pydub import AudioSegment
from src.streaming_response import stream_response_to_player

class FakeLLM:
    def __init__(self, chunks, delay_event=None):
        self.chunks = chunks
        self.delay_event = delay_event
        self.closed = False

    def generate_response_chunks(self, prompt):
        for chunk in self.chunks:
            if self.delay_event:
                self.delay_event.wait()
            yield chunk

    def close(self):
        self.closed = True

class FakeTTS:
    def synthesize(self, text, speed=None):
        return ([0.1] * 100, "phonemes")

class FakePlayer:
    def __init__(self, on_play_callback=None):
        self.played = []
        self.on_play_callback = on_play_callback

    def play_segment_blocking(self, segment):
        self.played.append(segment)
        if self.on_play_callback:
            self.on_play_callback(segment)

def test_streaming_ordering():
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
    assert len(player.played) == 3
    assert len(player.played[1]) == 1000

def test_streaming_concurrency():
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
    assert len(player.played) == 3
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
