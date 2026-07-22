"""Tests for voice layer functionality.

All tests run without hardware, models, or network — pure logic + mocks.
"""

import sys
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voice_layer import (
    split_sentences,
    discover_bleats,
    choose_bleat,
    numpy_to_segment,
    normalize_segment,
    synthesize_sentences,
    compose_with_bleat,
    create_spoken_response,
    KOKORO_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_wav(path: Path, duration_ms: int = 200, sample_rate: int = 24000) -> Path:
    """Create a minimal valid WAV file with silence."""
    n_samples = int(sample_rate * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _make_mock_tts(duration_ms: int = 100):
    """Return a mock TTS with a synthesize(text) method returning a short array."""
    n_samples = int(KOKORO_SAMPLE_RATE * duration_ms / 1000)
    audio = np.zeros(n_samples, dtype=np.float32)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = (audio, "phonemes")
    return mock_tts


# ===========================================================================
# split_sentences
# ===========================================================================

def test_split_sentences_single():
    """Single sentence preserved."""
    assert split_sentences("Xin chào.") == ["Xin chào."]


def test_split_sentences_multiple():
    """Multiple sentences split correctly."""
    assert split_sentences("Xin chào! Bạn khỏe không?") == [
        "Xin chào!",
        "Bạn khỏe không?",
    ]


def test_split_sentences_periods():
    """Period-delimited sentences."""
    assert split_sentences("Mình ổn. Còn bạn thì sao?") == [
        "Mình ổn.",
        "Còn bạn thì sao?",
    ]


def test_split_sentences_decimal():
    """Decimal numbers like 3.5 should not trigger a split."""
    result = split_sentences("Đây là phiên bản 3.5. Nó hoạt động tốt.")
    # "3.5" has a digit before the period — no split there.
    # The period after "tốt" has a non-digit before it — split.
    assert len(result) == 2
    assert "3.5" in result[0]


def test_split_sentences_no_punctuation():
    """Text without sentence-ending punctuation stays as one segment."""
    assert split_sentences("Không có dấu câu") == ["Không có dấu câu"]


def test_split_sentences_complex():
    """Mixed punctuation types."""
    result = split_sentences("Xin chào! Mình là một chú cừu thông minh. Bạn muốn hỏi gì?")
    assert result == [
        "Xin chào!",
        "Mình là một chú cừu thông minh.",
        "Bạn muốn hỏi gì?",
    ]


def test_split_sentences_empty():
    """Empty string returns empty list."""
    assert split_sentences("") == []


def test_split_sentences_whitespace_only():
    """Whitespace-only string returns empty list."""
    assert split_sentences("   \n\t  ") == []


def test_split_sentences_repeated_whitespace():
    """Repeated whitespace is collapsed."""
    result = split_sentences("Xin   chào!   Bạn   khỏe  không?")
    assert result == ["Xin chào!", "Bạn khỏe không?"]


# ===========================================================================
# discover_bleats
# ===========================================================================

def test_discover_bleats_missing_dir():
    """Non-existent directory returns empty list."""
    assert discover_bleats(Path("/nonexistent/path/bleats")) == []


def test_discover_bleats_empty_dir():
    """Empty directory returns empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_bleats(Path(tmp)) == []


def test_discover_bleats_with_wavs():
    """Directory with WAV files returns them sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_wav(d / "b.wav")
        _make_wav(d / "a.wav")
        (d / "readme.txt").write_text("ignore me")  # non-WAV

        result = discover_bleats(d)
        assert len(result) == 2
        assert result[0].name == "a.wav"
        assert result[1].name == "b.wav"


def test_discover_bleats_ignores_non_wav():
    """Non-WAV files are not returned."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "not_audio.mp3").write_bytes(b"\x00" * 10)
        (d / "data.json").write_text("{}")

        assert discover_bleats(d) == []


# ===========================================================================
# choose_bleat
# ===========================================================================

def test_choose_bleat_empty_list():
    """No bleats available returns None."""
    assert choose_bleat([]) is None


def test_choose_bleat_single():
    """Single bleat is always returned."""
    p = Path("assets/bleats/happy.wav")
    assert choose_bleat([p]) == p


def test_choose_bleat_multiple():
    """Multiple bleats — returns one of them."""
    paths = [Path(f"bleats/{i}.wav") for i in range(5)]
    result = choose_bleat(paths)
    assert result in paths


# ===========================================================================
# numpy_to_segment
# ===========================================================================

def test_numpy_to_segment_basic():
    """Float32 array converts to 16-bit mono AudioSegment."""
    audio = np.zeros(2400, dtype=np.float32)  # 0.1s at 24kHz
    seg = numpy_to_segment(audio, 24000)

    assert seg.frame_rate == 24000
    assert seg.channels == 1
    assert seg.sample_width == 2
    assert abs(len(seg) - 100) <= 1  # ~100ms


def test_numpy_to_segment_clipping():
    """Values outside [-1, 1] are clipped."""
    audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    seg = numpy_to_segment(audio, 24000)
    # Should not raise — clipping handles out-of-range values
    assert seg.sample_width == 2


# ===========================================================================
# compose_with_bleat
# ===========================================================================

def test_compose_empty():
    """Empty sentence list produces empty audio."""
    result = compose_with_bleat([], None)
    assert len(result) == 0


def test_compose_single_sentence_no_bleat():
    """Single sentence — no bleat inserted even if available."""
    seg = numpy_to_segment(np.zeros(2400, dtype=np.float32), 24000)

    with tempfile.TemporaryDirectory() as tmp:
        bleat = _make_wav(Path(tmp) / "bleat.wav")
        result = compose_with_bleat([seg], bleat)

        # Result should be just the sentence, no bleat
        assert abs(len(result) - 100) <= 2  # ~100ms sentence only


def test_compose_two_sentences_with_bleat():
    """Two sentences — bleat inserted after the first."""
    seg1 = numpy_to_segment(np.zeros(2400, dtype=np.float32), 24000)  # 100ms
    seg2 = numpy_to_segment(np.zeros(2400, dtype=np.float32), 24000)  # 100ms

    with tempfile.TemporaryDirectory() as tmp:
        bleat_path = _make_wav(Path(tmp) / "bleat.wav", duration_ms=200)
        result = compose_with_bleat([seg1, seg2], bleat_path, bleat_after_index=0)

        # seg1(100) + pause(100) + bleat(200) + pause(100) + pause(100) + seg2(100)
        # Total > 100 + 100 = 200ms (without bleat)
        assert len(result) > 200


def test_compose_max_one_bleat():
    """Only one bleat is inserted even with many sentences."""
    segs = [numpy_to_segment(np.zeros(2400, dtype=np.float32), 24000) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmp:
        bleat_path = _make_wav(Path(tmp) / "bleat.wav", duration_ms=200)
        result = compose_with_bleat(segs, bleat_path, bleat_after_index=0)

        # Compose again without bleat to compare
        result_no_bleat = compose_with_bleat(segs, None)

        # With bleat should be longer by roughly: bleat(200) + 2*pause(100) = 400ms
        diff_ms = len(result) - len(result_no_bleat)
        assert 300 < diff_ms < 500  # one bleat + pauses


def test_compose_no_bleat_file():
    """Missing bleat file causes graceful degradation."""
    seg = numpy_to_segment(np.zeros(2400, dtype=np.float32), 24000)
    result = compose_with_bleat([seg, seg], Path("/nonexistent/bleat.wav"))
    # Should still produce audio — just two sentences with a pause
    assert len(result) > 0


# ===========================================================================
# synthesize_sentences
# ===========================================================================

def test_synthesize_sentences_calls_tts():
    """TTS is called once per sentence."""
    mock_tts = _make_mock_tts()
    sentences = ["Câu một.", "Câu hai.", "Câu ba."]

    segments = synthesize_sentences(mock_tts, sentences)

    assert mock_tts.synthesize.call_count == 3
    assert len(segments) == 3


def test_synthesize_sentences_empty_audio_skipped():
    """Sentences producing empty audio are skipped."""
    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = [
        (np.zeros(2400, dtype=np.float32), "ok"),  # valid
        (np.array([], dtype=np.float32), ""),  # empty
        (np.zeros(2400, dtype=np.float32), "ok"),  # valid
    ]

    segments = synthesize_sentences(mock_tts, ["A.", "B.", "C."])
    assert len(segments) == 2  # B skipped


# ===========================================================================
# create_spoken_response (integration — uses mocks)
# ===========================================================================

def test_create_spoken_response_basic():
    """End-to-end: produces a final.wav file."""
    mock_tts = _make_mock_tts(duration_ms=50)

    with tempfile.TemporaryDirectory() as tmp:
        runtime = Path(tmp) / "runtime"
        bleats = Path(tmp) / "bleats"
        bleats.mkdir()
        _make_wav(bleats / "test.wav", duration_ms=100)

        result = create_spoken_response(
            response_text="Xin chào! Bạn khỏe không?",
            tts=mock_tts,
            bleats_dir=bleats,
            runtime_dir=runtime,
        )

        assert result.exists()
        assert result.name == "final.wav"


def test_create_spoken_response_no_bleats():
    """Works without any bleat files."""
    mock_tts = _make_mock_tts(duration_ms=50)

    with tempfile.TemporaryDirectory() as tmp:
        runtime = Path(tmp) / "runtime"
        # No bleats directory at all

        result = create_spoken_response(
            response_text="Chỉ có một câu.",
            tts=mock_tts,
            bleats_dir=Path(tmp) / "no_bleats",
            runtime_dir=runtime,
        )

        assert result.exists()


def test_create_spoken_response_empty_text():
    """Empty text raises ValueError."""
    mock_tts = _make_mock_tts()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            create_spoken_response(
                response_text="",
                tts=mock_tts,
                bleats_dir=Path(tmp),
                runtime_dir=Path(tmp) / "runtime",
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "no usable sentences" in str(e).lower()


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Fallback: run all test functions manually
        import traceback

        test_functions = [
            v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)
        ]

        passed = 0
        failed = 0
        for fn in test_functions:
            try:
                fn()
                passed += 1
                print(f"  PASS  {fn.__name__}")
            except Exception:
                failed += 1
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()

        print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests.")
        sys.exit(1 if failed else 0)
