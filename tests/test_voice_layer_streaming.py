import pytest
import random
from pathlib import Path
from pydub import AudioSegment
from unittest.mock import MagicMock
from src.voice_layer import (
    StreamingSentenceAssembler,
    split_sentences,
    synthesize_sentence,
    load_bleat_segments,
    build_inter_sentence_segment,
)

def test_assembler_basic():
    assembler = StreamingSentenceAssembler()
    assert assembler.feed("Hello world! How ") == ["Hello world!"]
    assert assembler.feed("are you? Fine.") == ["How are you?", "Fine."]
    assert assembler.finish() is None

def test_assembler_decimal():
    assembler = StreamingSentenceAssembler()
    assert assembler.feed("This is version 3.") == []
    assert assembler.feed("5 of the software.") == ["This is version 3.5 of the software."]
    assert assembler.finish() is None

def test_assembler_trailing_remainder():
    assembler = StreamingSentenceAssembler()
    assert assembler.feed("Unfinished sentence") == []
    assert assembler.finish() == "Unfinished sentence"

def test_split_sentences_matches():
    text = "Hello 3.5 world! This is a test. How are you?"
    assert split_sentences(text) == [
        "Hello 3.5 world!",
        "This is a test.",
        "How are you?"
    ]

def test_synthesize_sentence_empty():
    tts = MagicMock()
    tts.synthesize.return_value = ([], "phonemes")
    assert synthesize_sentence(tts, "hello") is None

def test_build_inter_sentence_segment_silence():
    # If no bleats or probability is 0, we expect silence of SILENCE_MS
    segment = build_inter_sentence_segment((), probability=0)
    # The SILENCE_MS constant determines length
    assert len(segment) > 0
