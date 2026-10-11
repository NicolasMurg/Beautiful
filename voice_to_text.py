#!/usr/bin/env python3
"""Local Whisper microphone speech-to-text on Jetson; Enter records, Q exits."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

from speech import WhisperCpp, add_speech_arguments, checked_run, record_audio, resolve_microphone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_speech_arguments(parser)
    parser.add_argument("--wav", type=Path, help="Transcribe an existing mono 16kHz PCM WAV")
    parser.add_argument("--once", action="store_true", help="Record/transcribe once, then exit")
    parser.add_argument("--list-devices", action="store_true", help="List ALSA capture devices")
    args = parser.parse_args()
    if not 1 <= args.record_seconds <= 20:
        parser.error("--record-seconds must be between 1 and 20")
    try:
        if args.list_devices:
            recorder = shutil.which("arecord")
            if recorder is None:
                raise FileNotFoundError("Install alsa-utils on the Jetson for microphone capture")
            print(checked_run([recorder, "-L"], 10))
            return
        recognizer = WhisperCpp(args.whisper_bin, args.whisper_model,
                               args.whisper_threads, args.speech_timeout)
        print(f"Local Whisper | CPU | model: {recognizer.model.name}", flush=True)
        if args.wav:
            print(json.dumps({"text": recognizer.transcribe(args.wav)}), flush=True)
            return
        mic_device = resolve_microphone()
        print(f"Microphone: {mic_device}", flush=True)
        while True:
            if not args.once:
                answer = input("Enter to record, or Q to quit: ").strip().lower()
                if answer == "q":
                    break
            with tempfile.TemporaryDirectory(prefix="voice-to-text-") as temporary:
                path = Path(temporary) / "microphone.wav"
                print(f"Recording {args.record_seconds}s; speak now...", flush=True)
                record_audio(path, args.record_seconds, mic_device)
                print("Transcribing...", flush=True)
                started = time.monotonic()
                text = recognizer.transcribe(path)
                print(json.dumps({"text": text,
                                  "transcription_seconds": time.monotonic() - started}), flush=True)
                if not text:
                    print("No transcript; check microphone level or speak closer.", flush=True)
            if args.once:
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.", flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Speech error: {exc}\n")


if __name__ == "__main__":
    main()
