#!/usr/bin/env python3
"""Pause the known stereo viewer, run the coach test, then restore that viewer."""

from jetson_paths import PROJECT_ROOT, RUNTIME_ROOT
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = RUNTIME_ROOT


def viewers():
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if str(ROOT/'live_dual_yolo_depth.py') in args:
                result.append((int(entry.name), args))
        except (OSError, UnicodeError):
            continue
    return result


def main():
    paused = []
    try:
        for pid, args in viewers():
            print(f'Pausing camera viewer {pid} to free memory for Qwen.', flush=True)
            os.kill(pid, signal.SIGINT)
            deadline = time.monotonic()+15
            while any(p == pid for p, _ in viewers()) and time.monotonic() < deadline:
                time.sleep(0.2)
            if any(p == pid for p, _ in viewers()):
                raise RuntimeError('Camera viewer did not stop; coach not started.')
            paused.append(args)
        code = subprocess.call([str(ROOT/'.venv/bin/python'), str(ROOT/'voice_coach_test.py'), *sys.argv[1:]],
                               env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
    finally:
        for args in paused:
            with (ROOT/'voice-coach/camera_restore.log').open('a') as log:
                process = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ, TORCHDYNAMO_DISABLE='1'),
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                           start_new_session=True)
            print(f'Restored prior camera command; new PID {process.pid}.', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
