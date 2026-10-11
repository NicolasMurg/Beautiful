#!/usr/bin/env python3
"""View two independent USB streams; timestamps describe arrival, not exposure."""

import argparse
from collections import deque
import fcntl
import struct
import threading
import time


FIRST = "/dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0"
SECOND = "/dev/v4l/by-id/usb-046d_1080P_Pro_Stream_C550F4FF-video-index0"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def positive_float(value):
    value = float(value)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def view_zoom(value):
    value = positive_float(value)
    if not 1 <= value <= 4:
        raise argparse.ArgumentTypeError("view zoom must be between 1 and 4")
    return value


def zoomed_view(cv2, frame, zoom):
    # Display-only crop: keep original capture frames and settings intact.
    height, width = frame.shape[:2]
    crop_width = max(1, round(width / zoom))
    crop_height = max(1, round(height / zoom))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(crop, (640, 480))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", "--brio", dest="brio", default=FIRST,
                        help="first capture device (--brio is a legacy alias)")
    parser.add_argument("--second", "--jieli", dest="jieli", default=SECOND,
                        help="second capture device (--jieli is a legacy alias)")
    parser.add_argument("--first-width", "--brio-width", dest="brio_width", type=positive_int, default=640)
    parser.add_argument("--first-height", "--brio-height", dest="brio_height", type=positive_int, default=480)
    parser.add_argument("--first-fps", "--brio-fps", dest="brio_fps", type=positive_float, default=25)
    parser.add_argument("--second-width", "--jieli-width", dest="jieli_width", type=positive_int, default=640)
    parser.add_argument("--second-height", "--jieli-height", dest="jieli_height", type=positive_int, default=480)
    parser.add_argument("--second-fps", "--jieli-fps", dest="jieli_fps", type=positive_float, default=30)
    parser.add_argument("--first-view-zoom", "--brio-view-zoom", dest="brio_view_zoom", type=view_zoom, default=1,
                        help="display-only centered crop, 1 to 4 (default: 1)")
    parser.add_argument("--second-view-zoom", "--jieli-view-zoom", dest="jieli_view_zoom", type=view_zoom, default=1,
                        help="display-only centered crop, 1 to 4 (default: 1)")
    parser.add_argument("--seconds", type=positive_float,
                        help="stop after this many seconds with both streams ready")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--startup-timeout", type=positive_float, default=10)
    parser.add_argument("--stale-timeout", type=positive_float, default=3)
    args = parser.parse_args()
    if args.no_display and args.seconds is None:
        parser.error("--no-display requires a finite --seconds duration")
    if args.brio == args.jieli:
        parser.error("the two camera devices must differ")
    return args


def disable_dynamic_frame_rate(path):
    # Brio auto-exposure may otherwise lower FPS to brighten a dark scene.
    control = bytearray(struct.pack("Ii", 0x009A0903, 0))
    with open(path, "rb", buffering=0) as device:
        fcntl.ioctl(device, 0xC008561C, control, True)  # VIDIOC_S_CTRL
        fcntl.ioctl(device, 0xC008561B, control, True)  # VIDIOC_G_CTRL
    if struct.unpack("Ii", control)[1] != 0:
        raise RuntimeError("Brio did not disable dynamic frame rate")
    print("Brio: dynamic frame rate disabled; automatic exposure remains enabled.", flush=True)


class Camera:
    def __init__(self, cv2, name, path, fourcc, width, height, fps, stop, history_size=0):
        self.cv2, self.name, self.path = cv2, name, path
        self.requested = fourcc, width, height, fps
        self.stop = stop
        self.lock = threading.Lock()
        self.history = deque(maxlen=history_size)
        self.frame = None
        self.count = 0
        self.first_arrival = self.last_arrival = None
        self.negotiated = None
        self.error = None
        self.recovering_since = None
        self.thread = threading.Thread(target=self.capture, name=name, daemon=True)

    def open_capture(self):
        cv2 = self.cv2
        cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {self.path}")
            fourcc, width, height, fps = self.requested
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_format = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
            negotiated = (actual_format, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          cap.get(cv2.CAP_PROP_FPS))
            with self.lock:
                self.negotiated = negotiated
            if actual_format != fourcc:
                raise RuntimeError(f"driver selected {actual_format}; required {fourcc}")
            if self.name == "Brio":
                disable_dynamic_frame_rate(self.path)
            print(f"{self.name}: requested {fourcc} {width}x{height} at {fps:g} fps; "
                  f"negotiated {negotiated[0]} {negotiated[1]}x{negotiated[2]} "
                  f"at {negotiated[3]:g} fps", flush=True)
            return cap
        except Exception:
            cap.release()
            raise

    def capture(self):
        cap = None
        recovery_deadline = None
        disconnects = 0
        try:
            cap = self.open_capture()
            while not self.stop.is_set():
                if cap is None:
                    if time.monotonic() >= recovery_deadline:
                        raise RuntimeError("USB reconnect timed out after 8 seconds; check cable/port")
                    if self.stop.wait(0.25):
                        break
                    try:
                        cap = self.open_capture()
                    except Exception:
                        continue
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    if self.stop.is_set():
                        break
                    cap.release()
                    cap = None
                    # Reconnecting Brio resets the privileged bandwidth workaround;
                    # the second camera can be reopened without reapplying it.
                    if self.name == "Brio":
                        raise RuntimeError("camera stopped delivering frames; restart the launcher")
                    if recovery_deadline is None:
                        disconnects += 1
                        if disconnects > 3:
                            raise RuntimeError("repeated USB dropouts (more than 3); check cable/port")
                        with self.lock:
                            self.history.clear()
                            self.frame = None
                            self.last_arrival = None
                            self.recovering_since = time.monotonic()
                        recovery_deadline = self.recovering_since + 8
                        print(f"{self.name}: feed lost; reconnecting ({disconnects}/3, up to 8 seconds).",
                              flush=True)
                    continue
                arrival = time.monotonic()
                if recovery_deadline is not None and arrival >= recovery_deadline:
                    raise RuntimeError("USB reconnect timed out after 8 seconds; check cable/port")
                with self.lock:
                    self.frame = frame
                    self.count += 1
                    self.last_arrival = arrival
                    self.recovering_since = None
                    if self.first_arrival is None:
                        self.first_arrival = arrival
                    if self.history.maxlen:
                        self.history.append((frame,self.count,self.first_arrival,arrival,None,None))
                if recovery_deadline is not None:
                    print(f"{self.name}: fresh frames resumed after reconnect.", flush=True)
                    recovery_deadline = None
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def history_snapshot(self):
        with self.lock:
            return list(self.history)

    def snapshot(self):
        with self.lock:
            return (self.frame, self.count, self.first_arrival,
                    self.last_arrival, self.error, self.recovering_since)


def run(args):
    # Delayed imports keep --help and argument errors usable without dependencies.
    import cv2
    import numpy as np

    stop = threading.Event()
    cameras = [
        Camera(cv2, "Jieli", args.brio, "MJPG", args.brio_width,
               args.brio_height, args.brio_fps, stop),
        Camera(cv2, "Pro Stream", args.jieli, "MJPG", args.jieli_width,
               args.jieli_height, args.jieli_fps, stop),
    ]
    window = "Dual cameras - independent streams (Q or Esc to exit)"
    displayed = False
    startup = time.monotonic()
    ready = report_at = healthy_since = None
    report_counts = [0, 0]
    rates = [0.0, 0.0]
    try:
        for camera in cameras:
            camera.thread.start()
        while True:
            now = time.monotonic()
            snapshots = [camera.snapshot() for camera in cameras]
            for camera, (_, _, _, arrival, error, recovering) in zip(cameras, snapshots):
                if error:
                    raise RuntimeError(f"{camera.name}: {error}")
                if recovering is not None and now - recovering >= 8:
                    raise RuntimeError(f"{camera.name}: USB reconnect timed out; check cable/port")
                if arrival is not None and now - arrival > args.stale_timeout:
                    raise RuntimeError(f"{camera.name}: no fresh frame for "
                                       f"{now - arrival:.1f} seconds")
            both_ready = all(item[0] is not None for item in snapshots)
            if not both_ready:
                healthy_since = None
            elif healthy_since is None:
                healthy_since = now
            if not both_ready and ready is None:
                if now - startup > args.startup_timeout:
                    missing = ", ".join(camera.name for camera, item in
                                        zip(cameras, snapshots) if item[0] is None)
                    raise RuntimeError(f"startup timed out waiting for {missing}")
                stop.wait(0.02)
                continue
            if ready is None:
                ready = report_at = now
                report_counts = [item[1] for item in snapshots]
                print("Both streams are delivering frames. Arrival times are "
                      "not synchronized exposure times.", flush=True)
                if not args.no_display:
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window, 1280, 480)
            if now - report_at >= 2:
                rates = [(item[1] - count) / (now - report_at)
                         for item, count in zip(snapshots, report_counts)]
                print(" | ".join(f"{camera.name}: {item[1]} frames, {rate:.1f} fps, "
                                 + (f"age {(now - item[3]) * 1000:.0f} ms"
                                  if item[3] is not None else "reconnecting")
                                 for camera, item, rate in zip(cameras, snapshots, rates)),
                      flush=True)
                report_counts = [item[1] for item in snapshots]
                report_at = now
            if not args.no_display:
                views = []
                zooms = (args.brio_view_zoom, args.jieli_view_zoom)
                for camera, item, rate, zoom in zip(cameras, snapshots, rates, zooms):
                    if item[0] is None:
                        view = np.zeros((480, 640, 3), dtype=np.uint8)
                        label = f"{camera.name}: USB disconnected; reconnecting..."
                    else:
                        view = zoomed_view(cv2, item[0], zoom)
                        label = (f"{camera.name}: {item[1]} frames | {rate:.1f} fps | "
                                 f"age {(now - item[3]) * 1000:.0f} ms | view {zoom:g}x")
                    cv2.rectangle(view, (0, 0), (640, 35), (0, 0, 0), -1)
                    cv2.putText(view, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    views.append(view)
                cv2.imshow(window, np.hstack(views))
                displayed = True
                if cv2.waitKey(1) & 255 in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if (args.seconds is not None and healthy_since is not None
                    and now - healthy_since >= args.seconds):
                break
            stop.wait(0.01)
    finally:
        stop.set()
        # V4L2 may block inside read until its own timeout. Give both workers a
        # shared bounded grace period; each worker owns and releases its handle.
        shutdown_deadline = time.monotonic() + 12
        for camera in cameras:
            if camera.thread.ident is not None:
                camera.thread.join(max(0, shutdown_deadline - time.monotonic()))
        if displayed:
            cv2.destroyAllWindows()
        for camera in cameras:
            _, count, first, last, _, _ = camera.snapshot()
            average = (count - 1) / (last - first) if count > 1 and last is not None and last > first else 0
            print(f"{camera.name}: total {count} frames; average {average:.1f} fps",
                  flush=True)
        stuck = [camera.name for camera in cameras if camera.thread.is_alive()]
        if stuck:
            raise RuntimeError("camera driver did not finish stopping: " +
                               ", ".join(stuck) + "; exit will close device handles")


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        raise SystemExit(f"Dual camera preview failed: {error}")
