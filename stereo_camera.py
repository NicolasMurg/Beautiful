"""Two USB camera capture/preview. Host grab times are not exposure timestamps."""

import argparse
import os
import sys
import time


class StereoCamera:
    def __init__(self, left, right, width=640, height=480, fps=15, max_grab_gap_ms=75):
        import cv2
        self.cv2 = cv2
        self.cameras = []
        self.max_grab_gap_ms = max_grab_gap_ms
        def identifier(source):
            return os.path.realpath(f'/dev/video{source}' if str(source).isdecimal()
                                    and sys.platform.startswith('linux') else str(source))
        if identifier(left) == identifier(right):
            raise ValueError("Left and right must be distinct physical capture devices")
        try:
            for source in (left, right):
                source = int(source) if str(source).isdecimal() else str(source)
                camera = cv2.VideoCapture(source, cv2.CAP_V4L2 if sys.platform.startswith('linux')
                                          else cv2.CAP_ANY)
                self.cameras.append(camera)
                if not camera.isOpened():
                    raise RuntimeError(f"Cannot open camera {source}")
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                camera.set(cv2.CAP_PROP_FPS, fps)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.close()
            raise

    def read(self):
        timestamps = []
        for camera in self.cameras:
            if not camera.grab():
                raise RuntimeError("A stereo camera stopped delivering frames")
            timestamps.append(time.monotonic())
        images = []
        for camera in self.cameras:
            ok, image = camera.retrieve()
            if not ok or image is None:
                raise RuntimeError("A stereo frame could not be retrieved")
            images.append(image)
        gap_ms = abs(timestamps[1] - timestamps[0]) * 1000
        metadata = {"host_grab_gap_ms": gap_ms, "received_monotonic_s": time.monotonic(),
                    "exposure_synchronized": False,
                    "pair_accepted": gap_ms <= self.max_grab_gap_ms}
        return images[0], images[1], metadata

    def close(self):
        for camera in self.cameras:
            camera.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-camera', required=True)
    parser.add_argument('--right-camera', required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    if min(args.width, args.height) <= 0 or args.frames < 0:
        parser.error('Positive resolution and nonnegative frame count are required')
    import cv2
    import numpy as np
    with StereoCamera(args.left_camera, args.right_camera, args.width, args.height) as camera:
        count = 0
        try:
            while args.frames == 0 or count < args.frames:
                left, right, metadata = camera.read()
                count += 1
                if count == 1 or count % 30 == 0:
                    print(f'Pair {count}: left {left.shape}, right {right.shape}; {metadata}', flush=True)
                if not args.no_display:
                    if left.shape != right.shape:
                        raise RuntimeError('Choose a capture resolution supported by both cameras')
                    cv2.imshow('Left | Right (unsynchronized USB)', np.hstack((left, right)))
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
