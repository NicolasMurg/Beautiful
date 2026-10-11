# Bottle reference calibration

`live_yolo.py` can adjust a metric depth model's scale using one operator-selected
bottle. Ordinary bottle detections never inherit the reference dimensions.
The feature is disabled until a profile is supplied and the operator confirms a
reference. It sends no actuator commands.

Create a local JSON profile with `name`, `height_m`, and `diameter_m`. Measure the
actual bottle; values are in meters. `reference_bottle.json` is ignored by Git,
so copy that file separately to the machine running the camera.

With camera intrinsics from `calibrate_camera.py`, run:

```bash
python3 live_yolo.py --depth-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf --reference-profile reference_bottle.json --calibration camera_calibration.yaml
```

Otherwise, keep the reference stationary and add
`--reference-distance-cm <measured-distance>` to the command in place of
`--calibration camera_calibration.yaml`. Measure the optical-axis distance from
the camera to the reference surface sampled by the model. A diagonal tape
distance to an off-center object is not the same as camera-axis depth.
Tape mode does not require camera intrinsics. Projected widths remain unknown
unless intrinsics are available.
If the reference moves, restore the measured position before confirming again,
or restart with its newly measured distance; pressing C does not update the
distance supplied on the command line.

1. Show the measured bottle fully, with the cap and bottom visible, upright and
   approximately parallel to the image plane. Avoid occlusion and frame edges.
2. Click its detection box. The selected tracked bottle is outlined in green.
3. Press **C** to confirm that this is the measured bottle in the required pose.
4. Press **R** to clear the reference. **Q** quits.

Height mode uses `Z_ref = fy * height_m / box_height_px` and a loose
height/diameter consistency check. Tape mode uses the supplied distance and
rejects appreciable reference-box movement. Each frame's scale is
`Z_ref / raw_model_Z_ref`; it multiplies that frame's metric object estimates.
Tracking loss, invalid depth, an inconsistent pose, frame-size changes, or
out-of-order frames clear the correction and require operator confirmation.
The profile is never selected automatically by object category.

The UI marks adjusted values as `ref est.`. JSON records retain `raw_depth_m`,
`depth_source`, the reference's track ID, and the scale/provenance for the frame.
Reference mode runs detection and depth on the same frame, overriding
`--depth-every` to 1; measure performance on the actual machine.

A single reference corrects only a scale discrepancy. It cannot repair
incorrect depth ratios elsewhere, guarantee tracker identity, or remove box,
pose, lens-distortion, or bottle-curvature errors. Height mode estimates the
bottle's projected plane distance; a depth model may instead predict its front
surface. Small differences matter at close range. The simple viewer does not
undistort images, so use a low-distortion camera/central placement and verify
accuracy in the intended working region. Resolution scaling is supported;
cropping, changed focus, or changed lenses require matching intrinsics.

Check reference and other objects at several independently measured distances
and poses, comparing raw and adjusted errors. Jetson performance, live camera
operation, and distance accuracy must be validated on the Jetson; numerical
software tests do not establish physical accuracy.
