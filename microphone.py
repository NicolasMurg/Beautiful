"""Discover ALSA capture inputs and verify one with a temporary PCM recording."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import wave


@dataclass(frozen=True)
class Microphone:
    device: str
    name: str
    usb_hint: bool = False


def parse_capture_devices(hardware_listing, pcm_listing):
    """Prefer recognizable USB/webcam inputs, then default/server routes, then others."""
    physical = []
    for line in hardware_listing.splitlines():
        match = re.match(r"^card (\d+): (\S+) \[([^\]]+)\], device (\d+): (.+)$", line)
        if match is None:
            continue
        card, card_id, card_name, device, device_name = match.groups()
        name = f"{card_name}: {device_name}"
        hint = any(word in f"{card_id} {name}".lower()
                   for word in ("usb", "webcam", "uvc", "logitech"))
        physical.append(Microphone(f"plughw:{card},{device}", name, hint))
    logical_names = [line.strip() for line in pcm_listing.splitlines()
                     if line and not line[0].isspace()]
    routes = [Microphone("default", "System default capture input")]
    for name in ("pulse", "pipewire"):
        if name in logical_names:
            routes.append(Microphone(name, f"{name} capture route"))
    other_routes = [Microphone(name, name) for name in logical_names
                    if name.startswith(("sysdefault:", "plughw:", "dsnoop:"))]
    ordered = ([mic for mic in physical if mic.usb_hint] + routes
               + [mic for mic in physical if not mic.usb_hint] + other_routes)
    unique = {}
    for mic in ordered:
        unique.setdefault(mic.device, mic)
    return list(unique.values())


def available_microphones(cancel=None):
    recorder = shutil.which("arecord")
    if recorder is None:
        raise FileNotFoundError("arecord is missing; install alsa-utils on the Jetson")
    listings = []
    environment = dict(os.environ, LC_ALL="C")
    for flag in ("-l", "-L"):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Microphone selection cancelled")
        try:
            result = subprocess.run([recorder, flag], stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", env=environment, timeout=5)
            listings.append(result.stdout if result.returncode == 0 else "")
        except subprocess.TimeoutExpired:
            listings.append("")
    return parse_capture_devices(*listings)


def validate_probe(path):
    """Require actual mono 16-bit/16kHz PCM data; silence is allowed in a quiet room."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("Capture input did not produce mono 16-bit 16kHz WAV")
        if audio.getnframes() < 8000 or len(audio.readframes(8000)) < 16000:
            raise ValueError("Capture input returned too little audio")


def choose_microphone(record_sample, cancel=None, report=None, probe_budget=12):
    """Try one-second captures, skipping busy/unsupported inputs with bounded waits.

    record_sample(path, device, timeout, cancel) must record a PCM WAV. The first
    successful capture is selected. A capture test does not prove audibility or
    identify which physical microphone the user intended.
    """
    if not math.isfinite(probe_budget) or probe_budget <= 0:
        raise ValueError("Microphone probe budget must be positive and finite")
    microphones = available_microphones(cancel)
    deadline = time.monotonic() + probe_budget
    failures = []
    with tempfile.TemporaryDirectory(prefix="microphone-probe-") as temporary:
        for index, microphone in enumerate(microphones[:8]):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Microphone selection cancelled")
            remaining = deadline - time.monotonic()
            if remaining < 1.5:
                break
            if report is not None:
                report(f"Checking microphone {microphone.device}: {microphone.name}")
            path = Path(temporary) / f"probe-{index}.wav"
            try:
                record_sample(path, microphone.device, min(3.0, remaining), cancel)
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Microphone selection cancelled")
                validate_probe(path)
            except (OSError, ValueError, RuntimeError, wave.Error) as exc:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Microphone selection cancelled") from exc
                failures.append(f"{microphone.device}: {str(exc)[:180]}")
                continue
            if report is not None:
                report(f"Selected microphone {microphone.device}: {microphone.name}")
            return microphone
    detail = "; ".join(failures[:3]) or "no capture input returned valid audio"
    raise RuntimeError(f"No usable microphone found ({detail}). Check the microphone connection, "
                       "capture permissions/level, and try again.")
