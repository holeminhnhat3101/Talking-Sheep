from __future__ import annotations

import select
import sys
import termios
import tty
from collections.abc import Callable, Sequence
from typing import TextIO


def read_terminal_key(stream: TextIO = sys.stdin) -> str:
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = stream.read(1)
        if first in ("\r", "\n"):
            return "ENTER"
        if first != "\x1b":
            return "OTHER"

        ready, _, _ = select.select([stream], [], [], 0.05)
        if not ready:
            return "ESC"

        suffix = stream.read(2)
        return {"[A": "UP", "[B": "DOWN"}.get(suffix, "OTHER")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def select_microphone_interactive(
    devices: Sequence[dict],
    *,
    read_key: Callable[[], str] | None = None,
    output: TextIO | None = None,
) -> int | None:
    if not devices:
        return None
    if len(devices) == 1:
        return int(devices[0]["index"])

    read_key = read_key or read_terminal_key
    output = output or sys.stdout
    selected = 0

    while True:
        output.write("\x1b[2J\x1b[H")
        output.write("Select microphone (↑/↓, Enter, Esc):\n\n")
        for position, device in enumerate(devices):
            marker = ">" if position == selected else " "
            output.write(
                f"{marker} [{device['index']}] {device['name']} | "
                f"{device['host_api']} | {device['channels']} ch | "
                f"{device['default_sample_rate']} Hz\n"
            )
        output.flush()

        key = read_key()
        if key == "UP":
            selected = (selected - 1) % len(devices)
        elif key == "DOWN":
            selected = (selected + 1) % len(devices)
        elif key == "ENTER":
            return int(devices[selected]["index"])
        elif key == "ESC":
            return None
