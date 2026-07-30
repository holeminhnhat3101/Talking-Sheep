import pytest
from unittest.mock import MagicMock, patch
from pydub import AudioSegment
from src.audio_player import AudioPlayer

@patch("pyaudio.PyAudio")
def test_audio_player_stream_reuse(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.return_value = mock_pa_instance
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    player = AudioPlayer()
    
    seg1 = AudioSegment(b"\x00" * 48000, sample_width=2, frame_rate=24000, channels=1)
    seg2 = AudioSegment(b"\x00" * 24000, sample_width=2, frame_rate=24000, channels=1)

    player.play_segment_blocking(seg1)
    player.play_segment_blocking(seg2)

    assert mock_pa_instance.open.call_count == 1
    assert mock_stream.write.call_count > 0

    player.close()
    mock_stream.close.assert_called_once()
    mock_pa_instance.terminate.assert_called_once()

@patch("pyaudio.PyAudio")
def test_audio_player_invalid_format(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.return_value = mock_pa_instance

    player = AudioPlayer()
    
    seg_wrong_rate = AudioSegment(b"\x00" * 100, sample_width=2, frame_rate=16000, channels=1)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_rate)

    seg_wrong_channels = AudioSegment(b"\x00" * 100, sample_width=2, frame_rate=24000, channels=2)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_channels)

    seg_wrong_width = AudioSegment(b"\x00" * 100, sample_width=1, frame_rate=24000, channels=1)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_width)

    player.close()
