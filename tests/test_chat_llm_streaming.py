import pytest
from unittest.mock import MagicMock, patch
from src.chat_llm import LLMChat, is_thinking, _StreamingResponseFilter, _SpacingNormalizer

def test_streaming_filter_think_tags():
    filt = _StreamingResponseFilter()
    assert filt.feed("hello <think>secret</think> world") == "hello  world"
    assert filt.finish() == ""

def test_streaming_filter_split_tags():
    filt = _StreamingResponseFilter()
    assert filt.feed("hello <th") == "hello "
    assert filt.feed("ink>secret</th") == ""
    assert filt.feed("ink> world") == " world"
    assert filt.finish() == ""

def test_streaming_filter_code_blocks():
    filt = _StreamingResponseFilter()
    assert filt.feed("hello ```code``` world") == "hello  world"
    assert filt.finish() == ""

def test_streaming_filter_unfinished():
    filt = _StreamingResponseFilter()
    assert filt.feed("hello <thi") == "hello "
    assert filt.finish() == "<thi"

def test_spacing_normalizer():
    norm = _SpacingNormalizer()
    # no whitespace at the beginning of the stream
    assert norm.normalize("   hello  ") == "hello "
    # leading whitespace in the next chunk is skipped because previous chunk ended with space
    assert norm.normalize("   world") == "world"
    # punctuation-adjacent chunks (previous chunk ended in 'd', not space, so leading space is kept)
    assert norm.normalize("   !") == " !"
    # trailing whitespace
    assert norm.normalize(" end   ") == " end "

@patch("src.chat_llm.ensure_model")
@patch("src.chat_llm.load_runtime_dependencies")
def test_generate_response_chunks(mock_load, mock_ensure):
    mock_llama = MagicMock()
    mock_load.return_value = lambda **kwargs: mock_llama
    mock_ensure.return_value = "/mock/model.gguf"

    chat = LLMChat()
    
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " <think>some thinking</think> "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
    ]
    mock_llama.create_chat_completion.return_value = chunks

    response_chunks = list(chat.generate_response_chunks("test prompt"))
    assert response_chunks == ["Hello", " ", "world"]
    assert list(chat.history) == [("test prompt", "Hello world")]

@patch("src.chat_llm.ensure_model")
@patch("src.chat_llm.load_runtime_dependencies")
def test_thinking_event_lifetime(mock_load, mock_ensure):
    mock_llama = MagicMock()
    mock_load.return_value = lambda **kwargs: mock_llama
    mock_ensure.return_value = "/mock/model.gguf"

    chat = LLMChat()
    
    def chunk_gen():
        assert is_thinking() is True
        yield {"choices": [{"delta": {"content": "hi"}}]}
        
    mock_llama.create_chat_completion.return_value = chunk_gen()
    
    assert is_thinking() is False
    gen = chat.generate_response_chunks("test prompt")
    assert is_thinking() is False
    item = next(gen)
    assert item == "hi"
    with pytest.raises(StopIteration):
        next(gen)
    assert is_thinking() is False
