#!/usr/bin/env python3
"""Connect the paired JBL and play a bounded test through its PipeWire sink."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

ADDRESS = os.environ.get("JBL_ADDRESS", "").upper()


def run(*args, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def find_sink():
    if not ADDRESS:
        raise RuntimeError('Set JBL_ADDRESS to your paired speaker Bluetooth address.')
    for obj in json.loads(run("pw-dump")):
        props = obj.get("info", {}).get("props", {})
        name = props.get("node.name", "")
        if (props.get("media.class") == "Audio/Sink"
                and name.startswith("bluez_output.")
                and ADDRESS.replace(":", "_") in name):
            return obj["id"], name, props.get("object.serial")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Inspect only; do not connect or play")
    parser.add_argument("--set-default", action="store_true", help="Also use JBL for desktop/app audio")
    args = parser.parse_args()
    if not ADDRESS:
        parser.error('Set JBL_ADDRESS to your paired speaker Bluetooth address.')
    info = run("bluetoothctl", "info", ADDRESS)
    print(info, flush=True)
    if args.status:
        print("JBL PipeWire sink:", find_sink(), flush=True)
        return
    if "Paired: yes" not in info:
        raise RuntimeError("Pair JBL Flip Essential 2 in Bluetooth Settings first.")
    if "Connected: yes" not in info:
        print("Connecting JBL; keep it powered on and disconnect other playback devices.", flush=True)
        output = run("bluetoothctl", "--timeout", "20", "connect", ADDRESS)
        print("\n".join(line for line in output.splitlines()
                        if ADDRESS in line or "Connection successful" in line
                        or "Failed to connect" in line), flush=True)
    deadline = time.monotonic() + 15
    sink = find_sink()
    while sink is None and time.monotonic() < deadline:
        time.sleep(0.5)
        sink = find_sink()
    if sink is None:
        raise RuntimeError("No JBL audio sink appeared. Check A2DP enablement and speaker connection.")
    node_id, name, serial = sink
    print(f"JBL output: {name} (id {node_id}, serial {serial})", flush=True)
    run("wpctl", "set-volume", str(node_id), "0.30")
    run("wpctl", "set-mute", str(node_id), "0")
    if args.set_default:
        run("wpctl", "set-default", str(node_id))
        print("JBL selected as default output.", flush=True)
    rate = 48000
    samples = array.array("h")
    # Brief silence lets Bluetooth wake, then left, right and both channels.
    segments = [(0, 0, 0, 1), (440, 1, 0, 2), (660, 0, 1, 2), (880, 1, 1, 2)]
    for frequency, left, right, duration in segments:
        for i in range(rate * duration):
            t = i / rate
            fade = max(0, min(1, t / 0.03, (duration - t) / 0.03))
            value = int(0.20 * 32767 * fade * math.sin(2 * math.pi * frequency * t))
            samples.extend((value * left, value * right))
    if sys.byteorder != "little":
        samples.byteswap()
    with tempfile.TemporaryDirectory(prefix="jbl-speaker-test-") as tmp:
        wav_path = Path(tmp) / "tones.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setparams((2, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(samples.tobytes())
        print("Playing 440 Hz, 660 Hz, then 880 Hz on JBL at 30% output volume.", flush=True)
        run("pw-play", "--target", str(serial), str(wav_path), timeout=20)
    if find_sink() is None or "Connected: yes" not in run("bluetoothctl", "info", ADDRESS):
        raise RuntimeError("JBL disconnected during playback.")
    print("Playback completed; JBL remains connected. Confirm that you heard all three tones.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Test failed: {error}", file=sys.stderr)
        sys.exit(1)
