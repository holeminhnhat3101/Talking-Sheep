"""Test cho chức năng lớp giọng nói.

Tất cả test chạy không cần phần cứng, model, hoặc mạng — logic thuần + mocks.
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
    numpy_to_segment,
    normalize_segment,
    synthesize_sentences,
    create_spoken_response,
    KOKORO_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
)


# ===========================================================================
# Helper
# ===========================================================================

def _make_wav(path: Path, duration_ms: int = 200, sample_rate: int = 24000) -> Path:
    """Tạo file WAV hợp lệ tối thiểu với im lặng."""
    n_samples = int(sample_rate * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _make_mock_tts(duration_ms: int = 100):
    """Trả về mock TTS với method synthesize(text) trả về mảng ngắn."""
    n_samples = int(KOKORO_SAMPLE_RATE * duration_ms / 1000)
    audio = np.zeros(n_samples, dtype=np.float32)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = (audio, "phonemes")
    return mock_tts


# ===========================================================================
# split_sentences
# ===========================================================================

def test_split_sentences_single():
    """Câu đơn được giữ nguyên."""
    assert split_sentences("Xin chào.") == ["Xin chào."]


def test_split_sentences_multiple():
    """Nhiều câu được tách đúng."""
    assert split_sentences("Xin chào! Bạn khỏe không?") == [
        "Xin chào!",
        "Bạn khỏe không?",
    ]


def test_split_sentences_periods():
    """Câu được phân tách bằng dấu chấm."""
    assert split_sentences("Mình ổn. Còn bạn thì sao?") == [
        "Mình ổn.",
        "Còn bạn thì sao?",
    ]


def test_split_sentences_decimal():
    """Số thập phân như 3.5 không nên kích hoạt tách câu."""
    result = split_sentences("Đây là phiên bản 3.5. Nó hoạt động tốt.")
    # "3.5" has a digit before the period — no split there.
    # The period after "tốt" has a non-digit before it — split.
    assert len(result) == 2
    assert "3.5" in result[0]


def test_split_sentences_no_punctuation():
    """Văn bản không có dấu câu kết thúc câu giữ nguyên như một đoạn."""
    assert split_sentences("Không có dấu câu") == ["Không có dấu câu"]


def test_split_sentences_complex():
    """Các loại dấu câu hỗn hợp."""
    result = split_sentences("Xin chào! Mình là một chú cừu thông minh. Bạn muốn hỏi gì?")
    assert result == [
        "Xin chào!",
        "Mình là một chú cừu thông minh.",
        "Bạn muốn hỏi gì?",
    ]


def test_split_sentences_empty():
    """Chuỗi rỗng trả về danh sách rỗng."""
    assert split_sentences("") == []


def test_split_sentences_whitespace_only():
    """Chuỗi chỉ có khoảng trắng trả về danh sách rỗng."""
    assert split_sentences("   \n\t  ") == []


def test_split_sentences_repeated_whitespace():
    """Khoảng trắng lặp lại được gộp lại."""
    result = split_sentences("Xin   chào!   Bạn   khỏe  không?")
    assert result == ["Xin chào!", "Bạn khỏe không?"]


# ===========================================================================
# discover_bleats
# ===========================================================================

def test_discover_bleats_missing_dir():
    """Thư mục không tồn tại trả về danh sách rỗng."""
    assert discover_bleats(Path("/nonexistent/path/bleats")) == []


def test_discover_bleats_empty_dir():
    """Thư mục rỗng trả về danh sách rỗng."""
    with tempfile.TemporaryDirectory() as tmp:
        assert discover_bleats(Path(tmp)) == []


def test_discover_bleats_with_wavs():
    """Thư mục có file WAV trả về chúng đã sắp xếp."""
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
    """File không phải WAV không được trả về."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "not_audio.mp3").write_bytes(b"\x00" * 10)
        (d / "data.json").write_text("{}")

        assert discover_bleats(d) == []


# ===========================================================================
# numpy_to_segment
# ===========================================================================

def test_numpy_to_segment_basic():
    """Mảng float32 chuyển đổi sang AudioSegment mono 16-bit."""
    audio = np.zeros(2400, dtype=np.float32)  # 0.1s at 24kHz
    seg = numpy_to_segment(audio, 24000)

    assert seg.frame_rate == 24000
    assert seg.channels == 1
    assert seg.sample_width == 2
    assert abs(len(seg) - 100) <= 1  # ~100ms


def test_numpy_to_segment_clipping():
    """Giá trị ngoài [-1, 1] được cắt."""
    audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    seg = numpy_to_segment(audio, 24000)
    # Should not raise — clipping handles out-of-range values
    assert seg.sample_width == 2


# ===========================================================================
# synthesize_sentences
# ===========================================================================

def test_synthesize_sentences_calls_tts():
    """TTS được gọi một lần cho mỗi câu."""
    mock_tts = _make_mock_tts()
    sentences = ["Câu một.", "Câu hai.", "Câu ba."]

    segments = synthesize_sentences(mock_tts, sentences)

    assert mock_tts.synthesize.call_count == 3
    assert len(segments) == 3


def test_synthesize_sentences_empty_audio_skipped():
    """Câu tạo ra audio rỗng bị bỏ qua."""
    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = [
        (np.zeros(2400, dtype=np.float32), "ok"),  # valid
        (np.array([], dtype=np.float32), ""),  # empty
        (np.zeros(2400, dtype=np.float32), "ok"),  # valid
    ]

    segments = synthesize_sentences(mock_tts, ["A.", "B.", "C."])
    assert len(segments) == 2  # B skipped


# ===========================================================================
# create_spoken_response (integration — sử dụng mocks)
# ===========================================================================

def test_create_spoken_response_basic():
    """End-to-end: tạo file final.wav."""
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
    """Hoạt động mà không cần file tiếng cừu nào."""
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
    """Văn bản rỗng raises ValueError."""
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
        # Fallback: chạy tất cả hàm test thủ công
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
