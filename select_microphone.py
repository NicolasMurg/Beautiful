#!/usr/bin/env python3
"""Automatically find a recording-capable microphone on Jetson; no Whisper needed."""

import argparse
from dataclasses import asdict
import json
import sys

from microphone import available_microphones, choose_microphone
from speech import record_audio


def probe_recording(path, device, timeout, cancel):
    record_audio(path, seconds=1, device=device, cancel=cancel, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List candidates without recording")
    args = parser.parse_args()
    try:
        if args.list:
            print(json.dumps([asdict(mic) for mic in available_microphones()], indent=2))
            return
        selected = choose_microphone(probe_recording,
                                     report=lambda text: print(text, file=sys.stderr, flush=True))
        print(json.dumps(asdict(selected)), flush=True)
    except KeyboardInterrupt:
        parser.exit(130, "\nMicrophone selection cancelled.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Microphone error: {exc}\n")


if __name__ == "__main__":
    main()
