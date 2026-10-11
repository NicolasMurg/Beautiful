#!/usr/bin/env python3
"""Temporary UVC bandwidth trial for this Jetson's Brio; no boot-time changes.

Run with system Python. `inspect` is read-only. `apply` and `restore` need root.
Only the Brio video control interface is rebound; its audio is left attached.
FIX_BANDWIDTH is useful for raw YUYV capture, not MJPEG.
"""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


USB = Path('/sys/bus/usb/devices')
DRIVER = Path('/sys/bus/usb/drivers/uvcvideo')
PARAMETER = Path('/sys/module/uvcvideo/parameters/quirks')
DEFAULT = 4294967295
FIX_BANDWIDTH = 128
SERIAL = '2514APE4NN78'
CAMERA = Path('/dev/v4l/by-id/usb-046d_Brio_101_2514APE4NN78-video-index0')
LOCK = Path('/run/jetson-brio-bandwidth.lock')


def find_brio():
    matches = []
    for device in USB.iterdir():
        try:
            identity = tuple((device / key).read_text().strip() for key in
                             ('idVendor', 'idProduct', 'serial'))
        except (FileNotFoundError, NotADirectoryError):
            continue
        if identity == ('046d', '094d', SERIAL):
            matches.append(device)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one Brio with serial {SERIAL}; found {len(matches)}')
    device = matches[0]
    interface = USB / (device.name + ':1.0')
    if not interface.exists():
        raise RuntimeError(f'Brio control interface is missing: {interface}')
    if (interface / 'bInterfaceClass').read_text().strip() != '0e' or \
            (interface / 'bInterfaceSubClass').read_text().strip() != '01':
        raise RuntimeError('Target interface is not a USB video control interface')
    return device, interface


def check_idle(device):
    nodes = [str(Path('/dev') / node.name)
             for node in Path('/sys/class/video4linux').glob('video*')
             if node.resolve().is_relative_to(device.resolve())]
    if nodes:
        result = subprocess.run(['/usr/bin/fuser', *nodes], capture_output=True,
                                text=True, timeout=5)
        if result.returncode == 0:
            raise RuntimeError('Brio is in use. Close its camera windows first. '
                               f'Process IDs: {result.stdout.strip()}')
        if result.returncode != 1 or result.stderr.strip():
            raise RuntimeError('Could not verify that the Brio is idle: '
                               + result.stderr.strip())


def write(path, value):
    path.write_text(str(value) + '\n')


def rebind(interface):
    binding = interface / 'driver'
    if binding.exists():
        if binding.resolve() != DRIVER.resolve():
            raise RuntimeError('Brio is attached to an unexpected driver')
        write(DRIVER / 'unbind', interface.name)
    write(DRIVER / 'bind', interface.name)
    if not binding.exists() or binding.resolve() != DRIVER.resolve():
        raise RuntimeError('Brio did not reattach to uvcvideo')


def wait_for_camera():
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if CAMERA.exists():
            return
        time.sleep(0.1)
    raise RuntimeError('Brio camera node did not return within 8 seconds')


def change(mode):
    if os.geteuid() != 0:
        raise RuntimeError('Administrator access is required; use sudo with /usr/bin/python3 -I')
    descriptor = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+') as lock:
        details = os.fstat(lock.fileno())
        if details.st_uid != 0 or not stat.S_ISREG(details.st_mode):
            raise RuntimeError('Unexpected driver-operation lock owner/type')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device, interface = find_brio()
        original = int(PARAMETER.read_text().strip())
        if original != DEFAULT:
            raise RuntimeError(f'Global quirks is already customized ({original}); leaving it unchanged')
        check_idle(device)
        try:
            # uvc_probe copies the parameter into the device during this rebind.
            try:
                write(PARAMETER, FIX_BANDWIDTH if mode == 'apply' else original)
                rebind(interface)
            finally:
                # Other cameras keep their normal defaults for future probes.
                write(PARAMETER, original)
            wait_for_camera()
        except BaseException:
            # A failed/interrupting application must not leave a detached camera.
            try:
                write(PARAMETER, original)
                rebind(interface)
                wait_for_camera()
                print('Restored Brio defaults after the failed operation.', file=sys.stderr)
            except BaseException as rollback_error:
                print(f'Rollback incomplete: {rollback_error}. Reconnect the Brio.', file=sys.stderr)
            raise
        print(('Applied temporary Brio-only bandwidth quirk 128; use raw YUYV.'
               if mode == 'apply' else 'Restored Brio default driver behavior.'), flush=True)
        print(f'Global UVC quirks restored to {original}; camera: {CAMERA.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'apply', 'restore'), nargs='?', default='inspect')
    args = parser.parse_args()
    if args.mode == 'inspect':
        device, interface = find_brio()
        print(f'Target: {device.name}, serial {SERIAL}; video control: {interface.name}')
        print(f'Global UVC quirks: {PARAMETER.read_text().strip()}')
        print('Apply: temporary per-Brio quirk 128 via rebind; restore: rebind with defaults.')
        print('Read-only inspection; no driver setting changed.')
    else:
        change(args.mode)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        sys.exit(130)
