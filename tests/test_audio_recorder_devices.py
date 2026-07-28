import numpy as np
import pytest
from src.audio_recorder import AudioRecorder

def test_default_alsa_device_is_virtual():
    device = {
        "name": "default",
        "channels": 128,
    }
    assert AudioRecorder._is_virtual_device(device)

def test_usb_hardware_device_is_not_virtual():
    device = {
        "name": "USB AUDIO DEVICE: Audio (hw:2,0)",
        "channels": 2,
    }
    assert not AudioRecorder._is_virtual_device(device)

def test_respeaker_name_detection():
    device = {
        "name": "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:1,0)",
    }
    assert AudioRecorder._is_respeaker(device)

def test_respeaker_scores_higher_than_default_device():
    default = {
        "index": 3,
        "name": "default",
        "channels": 128,
        "host_api": "ALSA",
    }
    respeaker = {
        "index": 1,
        "name": "ReSpeaker 4 Mic Array: USB Audio (hw:2,0)",
        "channels": 6,
        "host_api": "ALSA",
    }

    # Verify score ranking
    assert AudioRecorder._device_score(respeaker) > AudioRecorder._device_score(default)
    # Verify exact scores based on current implementation logic
    # respeaker: 0 + 1000 (respeaker) + 200 (hw:) + 100 (usb) = 1300
    # default: -1000 (virtual)
    assert AudioRecorder._device_score(respeaker) == 1300
    assert AudioRecorder._device_score(default) == -1000

def test_respeaker_auto_mode_uses_processed_channel_zero():
    # Use object.__new__ to bypass __init__ which opens PyAudio
    recorder = object.__new__(AudioRecorder)
    recorder.channel_mode = "auto"

    # 6-channel input where channel 4 has energy, but ReSpeaker should use channel 0
    samples = np.zeros((100, 6), dtype=np.float32)
    samples[:, 4] = 1.0

    selected = recorder._choose_channel(samples, is_respeaker=True)
    assert selected == 0

def test_generic_auto_mode_uses_highest_energy_channel():
    recorder = object.__new__(AudioRecorder)
    recorder.channel_mode = "auto"

    # 2-channel input where channel 1 has energy
    samples = np.zeros((100, 2), dtype=np.float32)
    samples[:, 1] = 1.0

    selected = recorder._choose_channel(samples, is_respeaker=False)
    assert selected == 1

def test_device_score_penalizes_high_channels():
    # A device that looks physical but has 128 channels should be penalized
    suspect = {
        "name": "USB Device with too many channels",
        "channels": 128,
    }
    # 0 + 100 (usb) - 500 (high channels) = -400
    assert AudioRecorder._device_score(suspect) == -400
