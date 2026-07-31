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
    
    # Updated to match new config: 48kHz, stereo, 16-bit
    seg1 = AudioSegment(b"\x00" * 96000, sample_width=2, frame_rate=48000, channels=2)
    seg2 = AudioSegment(b"\x00" * 48000, sample_width=2, frame_rate=48000, channels=2)

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
    
    # Updated to match new config: 48kHz, stereo, 16-bit
    seg_wrong_rate = AudioSegment(b"\x00" * 100, sample_width=2, frame_rate=16000, channels=2)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_rate)

    seg_wrong_channels = AudioSegment(b"\x00" * 100, sample_width=2, frame_rate=48000, channels=1)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_channels)

    seg_wrong_width = AudioSegment(b"\x00" * 100, sample_width=1, frame_rate=48000, channels=2)
    with pytest.raises(ValueError):
        player.play_segment_blocking(seg_wrong_width)

    player.close()

@patch("pyaudio.PyAudio")
def test_streaming_session(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.return_value = mock_pa_instance
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    player = AudioPlayer()
    
    # Start streaming session
    player.start_streaming_session()
    assert mock_pa_instance.open.call_count == 1
    
    # Write PCM data directly
    pcm_data = b"\x00" * 96000  # 48kHz, stereo, 16-bit
    player.write_pcm_blocking(pcm_data)
    assert mock_stream.write.call_count > 0
    
    # Write more data
    player.write_pcm_blocking(pcm_data)
    assert mock_stream.write.call_count > 1
    
    # End session
    player.end_streaming_session()
    assert mock_stream.stop_stream.called
    assert mock_stream.close.called
    
    player.close()

@patch("pyaudio.PyAudio")
def test_streaming_session_format_consistency(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.return_value = mock_pa_instance
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    player = AudioPlayer()
    
    player.start_streaming_session()
    
    # Verify stream opened with correct format
    open_kwargs = mock_pa_instance.open.call_args[1]
    assert open_kwargs['rate'] == 48000
    assert open_kwargs['channels'] == 2
    assert open_kwargs['format'] == mock_pa_instance.get_format_from_width(2)
    
    player.end_streaming_session()
    player.close()

@patch("pyaudio.PyAudio")
def test_streaming_session_write_without_start(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.return_value = mock_pa_instance

    player = AudioPlayer()
    
    with pytest.raises(RuntimeError, match="Not in streaming session"):
        player.write_pcm_blocking(b"\x00" * 100)
    
    player.close()
