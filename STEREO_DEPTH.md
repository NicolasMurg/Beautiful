# Stereo distance and projected object size

The default physical lens-center spacing is **5 inches = 0.127 m**. The two
cameras face the same scene and must stay fixed after calibration. Stereo mode
uses geometric correspondence and does not require a reference bottle or a
monocular depth model.

## 1. Identify and preview the two physical cameras on the Jetson

Stop other camera applications. Run `v4l2-ctl --list-devices` and inspect
`v4l2-ctl -d DEVICE --list-formats-ext` for each physical camera's capture node.
If needed, install the device-list utility with `sudo apt install v4l-utils`.
One webcam may expose multiple nodes, including metadata nodes. Prefer stable
`/dev/v4l/by-id/` or `/dev/v4l/by-path/` capture paths once identified.

In the shell running your existing OpenCV/YOLO environment, set actual paths:

```bash
LEFT_CAMERA=/dev/video0
RIGHT_CAMERA=/dev/video2
python3 stereo_camera.py --left-camera "$LEFT_CAMERA" --right-camera "$RIGHT_CAMERA"
```

Those paths are examples; do not assume they identify your cameras. Confirm
physical left/right by covering each lens. Both views must show the object and
calibration board. The requested mode is MJPEG, 640 x 480, 15 FPS. Verify actual
capture output. USB bandwidth, camera settings and synchronization must be
checked on the actual Jetson. `--no-display --frames 30` checks capture in a
terminal. No GPU packages need replacing for stereo calibration/matching.

## 2. Calibrate the fixed rig

Print the included `stereo_checkerboard.svg` at **100% / actual size** on a flat
rigid surface. It has 9 x 6 inner corners and nominal 18 mm squares. Measure a
printed square; the measured length, rather than the nominal length, goes in
`--square-mm`. You can generate another board using `make_checkerboard.py`.

```bash
python3 calibrate_stereo.py --left-camera "$LEFT_CAMERA" --right-camera "$RIGHT_CAMERA" --square-mm 18 --baseline-inches 5
```

Replace 18 if the printed measurement differs. Hold the board still, fully
visible in both cameras. Move it through varied positions, distances and tilts
within the shared view; press SPACE to save each stationary paired view. Save
at least 15 views, then press C. Calibration writes `stereo_calibration.npz`.
Calibration error, vertical correspondence error and agreement with the measured
baseline are checked; these are internal checks, not physical validation.

Calibration stores intrinsics, distortion, relative rotation/translation and
capture size. Its checkerboard-derived baseline must agree with 0.127 m within
5%; otherwise the code rejects it. Runtime preserves calibrated orientation and
uses the measured 0.127 m spacing to establish the metric translation scale.
Recalibrate after any change to mounting, focus, lens or capture resolution.
Passing a new baseline argument does not fix calibration from a different rig.

## 3. Run YOLO with stereo depth/size

Using your existing YOLO weights and environment:

```bash
python3 live_yolo.py --camera "$LEFT_CAMERA" --right-camera "$RIGHT_CAMERA" --stereo-calibration stereo_calibration.npz --baseline-inches 5
```

Add `--model PATH` if the weights are not `yolo26n.pt` beside the script. The
YOLO device defaults to GPU 0; `--device cpu` selects CPU explicitly. Stereo
matching currently uses CPU OpenCV SGBM. Do not supply `--depth-model` or
`--reference-profile` in stereo mode. A standalone entry point also exists:

```bash
python3 live_yolo_stereo.py --left-camera "$LEFT_CAMERA" --right-camera "$RIGHT_CAMERA" --stereo-calibration stereo_calibration.npz --baseline-inches 5
```

The display shows rectified left/right views, detected objects, distance Z in
meters and approximate bounding-box width/height in centimeters. Q quits.
JSON records printed once per second contain track/frame IDs, meter units,
valid-depth fraction, depth spread, capture status and explicit measurement
sources. `--no-display --frames 30` provides a numerical check.

## What the calculation means

Rectified stereo uses
`Z = fx * baseline_m / (disparity_px - (cx_left - cx_right))`.
Matching uses image texture, rather than subtracting the centers of two YOLO
boxes. Reverse matching rejects inconsistent/occluded correspondences; invalid
or missing depth stays unknown. No monocular fallback is applied silently.

YOLO runs on the rectified left image used by the matcher. Each object's central
box region supplies a robust median Z. Too few valid pixels or a large depth
spread yields unknown measurements. The visible projected estimates are
`width = bbox_width_px * Z / fx` and `height = bbox_height_px * Z / fy`.
Partial boxes have unknown size. All objects are measured independently; two
different bottles do not inherit a shared size.

These are axis-aligned box estimates at a representative visible depth, not full
3D dimensions, hidden thickness, volume, or a calibrated hand opening. Background
inside the box, loose detections, object rotation and surface curvature can bias
them. Transparent/reflective or textureless objects can have missing or erroneous
depth. Segmentation and independently checked surface geometry are future work.

## Capture and working range

Two ordinary USB webcams are **not assumed to have synchronized exposure**. The
capture tool grabs both before retrieving either, timestamps host grab completion,
and rejects pairs with a host grab gap over 75 ms. This checks host timing only;
it cannot prove that exposures coincide or detect every stale driver buffer.
For this initial backend, test static/slow objects and keep the board stationary
when saving calibration views. Moving-scene accuracy requires actual capture
synchronization or a validated timestamp-pairing backend.

The default disparity search is 192 pixels. The approximate near limit imposed
by this search is `fx * 0.127 / 191`; the real usable range also depends on view
overlap, texture, matcher margins and camera behavior. `--stereo-disparities`
accepts a positive multiple of 16 smaller than image width. Increasing it may
help nearer objects but costs computation and image coverage; it does not
guarantee valid close-range results.

Before using measurements for physical decisions, compare distance and projected
width/height against independent measurements at multiple distances and poses.
Record missing depth as failure rather than zero. Measure throughput and frame
age on the Jetson. Local synthetic tests verify software mathematics and matching
only; this Windows workspace cannot establish hardware accuracy. No hardware
actuation is included in the stereo viewer.

Primary API references: [OpenCV calibration/rectification](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
and [OpenCV stereo depth](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html).
