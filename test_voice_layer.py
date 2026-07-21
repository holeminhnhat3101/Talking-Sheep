"""Tests for voice layer functionality."""
import pytest
from voice_layer import split_sentences, choose_bleat_position, choose_bleat


def test_split_sentences_single():
    """Test splitting a single sentence."""
    text = "Xin chào."
    result = split_sentences(text)
    assert result == ["Xin chào."]


def test_split_sentences_multiple():
    """Test splitting multiple sentences."""
    text = "Xin chào! Bạn khỏe không?"
    result = split_sentences(text)
    assert result == ["Xin chào!", "Bạn khỏe không?"]


def test_split_sentences_periods():
    """Test splitting with periods."""
    text = "Mình ổn. Còn bạn thì sao?"
    result = split_sentences(text)
    assert result == ["Mình ổn.", "Còn bạn thì sao?"]


def test_split_sentences_decimal():
    """Test that decimal numbers don't cause incorrect splits."""
    text = "Đây là phiên bản 3.5. Nó hoạt động tốt."
    result = split_sentences(text)
    # Should split at the period after "tốt", not after "3.5"
    assert len(result) == 2
    assert "3.5" in result[0]


def test_split_sentences_no_punctuation():
    """Test text without punctuation remains one segment."""
    text = "Không có dấu câu"
    result = split_sentences(text)
    assert result == ["Không có dấu câu"]


def test_split_sentences_complex():
    """Test complex sentence with multiple punctuation types."""
    text = "Xin chào! Mình là một chú cừu thông minh. Bạn muốn hỏi gì?"
    result = split_sentences(text)
    assert result == [
        "Xin chào!",
        "Mình là một chú cừu thông minh.",
        "Bạn muốn hỏi gì?",
    ]


def test_choose_bleat_position_single_sentence():
    """Test that no bleat is inserted with single sentence."""
    sentences = ["Xin chào."]
    result = choose_bleat_position(sentences)
    assert result is None


def test_choose_bleat_position_multiple_sentences():
    """Test bleat position with multiple sentences."""
    sentences = ["Xin chào!", "Bạn khỏe không?"]
    result = choose_bleat_position(sentences, probability=1.0)
    assert result == 1


def test_choose_bleat_position_probability():
    """Test that probability affects bleat insertion."""
    sentences = ["Xin chào!", "Bạn khỏe không?", "Mình ổn."]
    # With probability 0, should never insert
    result = choose_bleat_position(sentences, probability=0.0)
    assert result is None


def test_choose_bleat_happy():
    """Test bleat selection for happy sentences."""
    sentence = "Tuyệt vời!"
    result = choose_bleat(sentence)
    assert result.name == "happy.wav"


def test_choose_bleat_confused():
    """Test bleat selection for confused/question sentences."""
    sentence = "Bạn nói gì?"
    result = choose_bleat(sentence)
    assert result.name == "confused.wav"


def test_choose_bleat_neutral():
    """Test bleat selection for neutral sentences."""
    sentence = "Mình ổn."
    result = choose_bleat(sentence)
    # Should be either short.wav or happy.wav
    assert result.name in ["short.wav", "happy.wav"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
