import argparse
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import microphone
import speech


HARDWARE = """**** List of CAPTURE Hardware Devices ****
card 0: APE [NVIDIA Tegra APE], device 0: Internal [Internal]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: Webcam [USB Webcam], device 1: USB Audio [USB Audio]
  Subdevices: 0/1
card 3: Other [Other input], device 0: Analog [Analog]
"""
PCMS = """null
    Discard all samples (playback) or generate zero samples (capture)
default
    Default capture device
pulse
    PulseAudio server
plughw:2,1
    An alias of the webcam
sysdefault:CARD=Other
    Another capture route
"""


def write_probe(path, frames=16000, channels=1, rate=16000):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * frames * channels)


def test_capture_list_prefers_usb_and_deduplicates_routes():
    microphones = microphone.parse_capture_devices(HARDWARE, PCMS)
    devices = [mic.device for mic in microphones]
    assert devices[:3] == ["plughw:2,1", "default", "pulse"]
    assert devices.count("plughw:2,1") == 1
    assert "plughw:0,0" in devices and "plughw:3,0" in devices
    assert "null" not in devices
    assert microphones[0].usb_hint


def test_discovery_uses_capture_lists_and_fixed_locale(monkeypatch):
    calls = []
    monkeypatch.setattr(microphone.shutil, "which", lambda name: "/usr/bin/arecord")
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=HARDWARE if command[-1] == "-l" else PCMS)
    monkeypatch.setattr(microphone.subprocess, "run", fake_run)
    assert microphone.available_microphones()[0].device == "plughw:2,1"
    assert [command[-1] for command, _ in calls] == ["-l", "-L"]
    assert all(kwargs["env"]["LC_ALL"] == "C" for _, kwargs in calls)
    assert all(kwargs["timeout"] == 5 for _, kwargs in calls)


def test_missing_recorder_has_actionable_error(monkeypatch):
    monkeypatch.setattr(microphone.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="alsa-utils"):
        microphone.available_microphones()


def candidates(monkeypatch, devices=None):
    items = devices or ["plughw:2,1", "default", "plughw:3,0"]
    monkeypatch.setattr(microphone, "available_microphones",
                        lambda cancel=None: [microphone.Microphone(name, name) for name in items])


def test_selection_skips_busy_device_and_accepts_valid_quiet_capture(monkeypatch):
    candidates(monkeypatch)
    calls, paths = [], []
    def record(path, device, timeout, cancel):
        calls.append(device)
        paths.append(path)
        assert timeout <= 3
        if device == "plughw:2,1":
            raise RuntimeError("Device or resource busy")
        write_probe(path)  # Quiet room is not evidence that this isn't a microphone.
    selected = microphone.choose_microphone(record)
    assert selected.device == "default"
    assert calls == ["plughw:2,1", "default"]
    assert all(not path.exists() for path in paths)  # No retained microphone recordings.


def test_selection_rejects_wrong_format_and_empty_capture(monkeypatch):
    candidates(monkeypatch)
    def record(path, device, timeout, cancel):
        if device == "plughw:2,1":
            write_probe(path, channels=2)
        elif device == "default":
            write_probe(path, frames=0)
        else:
            write_probe(path)
    assert microphone.choose_microphone(record).device == "plughw:3,0"


def test_no_usable_microphone_reports_failure_instead_of_guessing(monkeypatch):
    candidates(monkeypatch)
    def record(*args):
        raise RuntimeError("capture unavailable")
    with pytest.raises(RuntimeError, match="No usable microphone found"):
        microphone.choose_microphone(record)


def test_probe_limits_and_expiry(monkeypatch):
    candidates(monkeypatch, [f"plughw:{index},0" for index in range(20)])
    attempts = []
    def record(path, device, timeout, cancel):
        attempts.append(device)
        raise RuntimeError("busy")
    with pytest.raises(RuntimeError):
        microphone.choose_microphone(record)
    assert len(attempts) == 8
    clock = iter([0, 0, 13])
    monkeypatch.setattr(microphone.time, "monotonic", lambda: next(clock))
    attempts.clear()
    with pytest.raises(RuntimeError):
        microphone.choose_microphone(record)
    assert len(attempts) == 1


def test_cancellation_does_not_try_another_microphone(monkeypatch):
    candidates(monkeypatch)
    cancel = threading.Event()
    attempts = []
    def record(path, device, timeout, event):
        attempts.append(device)
        event.set()
        write_probe(path)
    with pytest.raises(RuntimeError, match="cancelled"):
        microphone.choose_microphone(record, cancel)
    assert len(attempts) == 1


def test_speech_commands_do_not_require_or_expose_a_manual_device():
    parser = argparse.ArgumentParser()
    speech.add_speech_arguments(parser)
    args = parser.parse_args([])
    assert not hasattr(args, "mic_device")
    assert "--mic-device" not in parser.format_help()


def test_worker_selects_once_and_reselects_after_capture_failure(monkeypatch):
    selections, recordings = [], []
    def resolve(cancel=None):
        selections.append(cancel)
        return "plughw:2,1"
    fail_next = [False]
    def record(path, seconds, device, cancel=None):
        recordings.append(device)
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("microphone disconnected")
        write_probe(path)
    monkeypatch.setattr(speech, "resolve_microphone", resolve)
    monkeypatch.setattr(speech, "record_audio", record)
    worker = speech.SpeechWorker(SimpleNamespace(transcribe=lambda *args: "hello"))
    try:
        for _ in range(2):
            assert worker.request()
            worker._thread.join(timeout=2)
            assert worker.poll().text == "hello"
        assert len(selections) == 1
        assert recordings == ["plughw:2,1", "plughw:2,1"]
        fail_next[0] = True
        assert worker.request()
        worker._thread.join(timeout=2)
        assert "disconnected" in worker.poll().error
        assert worker.device == "auto"
        assert worker.request()
        worker._thread.join(timeout=2)
        assert worker.poll().text == "hello"
        assert len(selections) == 2
    finally:
        worker.close()
