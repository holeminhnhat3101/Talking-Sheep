import pytest
from unittest.mock import MagicMock, patch
from src.talking_sheep_voice import (
    resolve_startup_input_selector,
    run_once,
)

def test_resolve_startup_input_selector_bypass():
    recorder = MagicMock()
    selector, should_exit = resolve_startup_input_selector(
        recorder,
        configured_selector=3,
        interactive=True,
    )
    assert selector == 3
    assert should_exit is False
    recorder.list_selectable_input_devices.assert_not_called()

def test_resolve_startup_input_selector_one_device():
    recorder = MagicMock()
    recorder.list_selectable_input_devices.return_value = [{"index": 1, "name": "Mic"}]
    selector, should_exit = resolve_startup_input_selector(
        recorder,
        configured_selector=None,
        interactive=True,
    )
    assert selector == 1
    assert should_exit is False

def test_resolve_startup_input_selector_non_interactive():
    recorder = MagicMock()
    recorder.list_selectable_input_devices.return_value = [
        {"index": 1, "name": "Mic1"},
        {"index": 2, "name": "Mic2"},
    ]
    selector, should_exit = resolve_startup_input_selector(
        recorder,
        configured_selector=None,
        interactive=False,
    )
    assert selector is None
    assert should_exit is False

def test_resolve_startup_input_selector_escape():
    recorder = MagicMock()
    recorder.list_selectable_input_devices.return_value = [
        {"index": 1, "name": "Mic1"},
        {"index": 2, "name": "Mic2"},
    ]
    menu_selector = MagicMock(return_value=None)
    selector, should_exit = resolve_startup_input_selector(
        recorder,
        configured_selector=None,
        interactive=True,
        menu_selector=menu_selector,
    )
    assert selector is None
    assert should_exit is True

@patch("src.talking_sheep_voice.stream_response_to_player")
def test_run_once_integration(mock_stream):
    recorder = MagicMock()
    recorder.capture_utterance_stream.return_value = ("/path/to/input.wav", 12345.67)
    
    stt = MagicMock()
    stt.finish_utterance.return_value = "Hello world."
    
    llm = MagicMock()
    tts = MagicMock()
    player = MagicMock()
    bleat_segments = (MagicMock(),)
    
    mock_stream.return_value = True
    
    completed = run_once(
        recorder,
        stt,
        llm,
        tts,
        player,
        runtime_dir=MagicMock(),
        bleat_segments=bleat_segments,
        silence_threshold=0.1,
        silence_duration=1.0,
    )
    
    assert completed is True
    mock_stream.assert_called_once_with(
        prompt="Hello world.",
        llm=llm,
        tts=tts,
        player=player,
        bleat_segments=bleat_segments,
        speech_end_time=12345.67,
    )
    player.play_blocking.assert_not_called()
