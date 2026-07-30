from io import StringIO
from src.audio_recorder import AudioRecorder
from src.microphone_menu import select_microphone_interactive


def test_list_selectable_input_devices_prefers_physical(monkeypatch):
    recorder = object.__new__(AudioRecorder)
    monkeypatch.setattr(
        recorder,
        "list_input_devices",
        lambda: [
            {"index": 0, "name": "default", "channels": 128,
             "default_sample_rate": 44100, "host_api": "ALSA"},
            {"index": 1, "name": "ReSpeaker 4 Mic Array", "channels": 6,
             "default_sample_rate": 16000, "host_api": "ALSA"},
        ],
    )

    assert [d["index"] for d in recorder.list_selectable_input_devices()] == [1]


def test_list_selectable_input_devices_excludes_virtual_only_list(monkeypatch):
    recorder = object.__new__(AudioRecorder)
    devices = [
        {"index": 0, "name": "default", "channels": 128,
         "default_sample_rate": 44100, "host_api": "ALSA"}
    ]
    monkeypatch.setattr(recorder, "list_input_devices", lambda: devices)

    assert recorder.list_selectable_input_devices() == []


DEVICES = [
    {"index": 1, "name": "USB Microphone", "host_api": "ALSA",
     "channels": 1, "default_sample_rate": 48000},
    {"index": 3, "name": "ReSpeaker 4 Mic Array", "host_api": "ALSA",
     "channels": 6, "default_sample_rate": 16000},
]


def key_reader(*keys):
    iterator = iter(keys)
    return lambda: next(iterator)


def test_menu_moves_down_and_confirms():
    selected = select_microphone_interactive(
        DEVICES,
        read_key=key_reader("DOWN", "ENTER"),
        output=StringIO(),
    )
    assert selected == 3


def test_menu_wraps_up_from_first_item():
    selected = select_microphone_interactive(
        DEVICES,
        read_key=key_reader("UP", "ENTER"),
        output=StringIO(),
    )
    assert selected == 3


def test_menu_escape_cancels():
    assert select_microphone_interactive(
        DEVICES,
        read_key=key_reader("ESC"),
        output=StringIO(),
    ) is None


def test_menu_auto_selects_single_device_without_reading_key():
    assert select_microphone_interactive(
        DEVICES[:1],
        read_key=lambda: (_ for _ in ()).throw(AssertionError("must not read")),
        output=StringIO(),
    ) == 1
