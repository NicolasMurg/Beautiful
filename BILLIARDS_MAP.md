# Camera measurements to a virtual pool table

Use [TABLE_WORKFLOW.md](TABLE_WORKFLOW.md) for runnable `calibrate_table.py`,
`capture_ball_measurements.py`, and `map_table_state.py` commands. This document
describes the underlying mapping interfaces and coordinate math.

Standalone alignment uses four reference balls, one near each corner, with a
three-inch (7.62 cm) clear gap from each ball's surface to both adjacent inner
cushion noses. Once saved, the table profile stays aligned after those balls
are removed for play; mapping uses the stored transform, not reference tracking.

The virtual map is the intermediate representation used for planning. It contains
observed ball centers in **centimeters**, the calibrated pocket openings and
cushion segments, identities, uncertainty, and review status. Its origin is the
top-left inner cushion intersection; x increases toward top-right and y toward
bottom-left. Both the overhead display and shot planner consume this map.

## Required measurements

An object's distance from a camera does not identify a unique table position.
The table mapper needs its image position as well as the camera's calibrated
intrinsics and pose relative to the table. Two scalar camera ranges by themselves still
leave geometric ambiguity. The existing stereo pipeline already combines both
camera images and supplies depth aligned with the rectified left image.

Ball-to-ball spacing alone needs no table pose: image positions plus metric
depth and matching intrinsics reconstruct centers in the camera frame. Their
separation is unchanged by the camera's position or orientation. For a side
mount, there is no need to measure the camera's setback from the table edge.
The four-reference calibration estimates table alignment when pocket-relative
positions are needed for shot planning. In the standalone script's default ball
mode, the known center inset is 7.62 cm plus the measured ball radius. The fit
also removes the ball centers' height above the felt so the saved origin remains
on the felt. The integrated live application's flat-marker workflow is also
supported.

For the existing live application, each scan proceeds through:

1. `BallDetector` proposes a ball's pixel center and image radius.
2. `TableCalibration.localize_ball` samples valid metric stereo depth and
   reconstructs its center in the table coordinate frame.
3. `build_table_map` collects those localizations, calibrated pockets and rails.
   It preserves unidentified balls and reports failed localizations as `unmapped`.
4. The operator checks that every ball currently on the table is represented,
   including potential obstructions hidden from either camera.
5. `planner_state_from_map` extracts the reviewed metric layout for `plan_shots`.

Identification and position reconstruction are separate. A tracker key is not a
ball number. No failed depth reading is silently interpreted as a pocketed ball.
Pockets and rails come from the measured table profile; raw YOLO part boxes do
not establish their physical geometry.

Pocket recognition is optional. For a standard six-pocket table, measured
cushion corners, table dimensions, and mouth widths supply the pocket geometry;
individual measured overrides are supported. These are calibrated references,
not detections. The map contains only the current physical scene. It does not
record pocketing history or player ownership; each solids/stripes request selects
targets while retaining all other balls as obstacles.

## Coordinate conversion

Let `K` be the camera's intrinsic matrix and `q = inverse(K) * [u, v, 1]` the
viewing ray for a pixel in the profile's rectified image. There are two distinct
distance conventions:

- **camera_z:** `P_measured = q * (distance_m / q.z)`.
- **range:** `P_measured = normalize(q) * distance_m`.

For a visible-surface measurement at the projected ball center, the mapper adds
one physical ball radius along that ray to estimate the sphere center. For a
measurement already referring to the ball center, it adds nothing.

The calibrated origin and rotation are stored in meters. The rigid conversion is
`P_table = transpose(rotation) * (P_center - origin)`. Its first two components,
multiplied by 100, give table x/y in centimeters. The third component checks that
the center is at the ball's expected height above the felt. No scale is fitted to
make arbitrary relative depth appear metric.

This uses the same transform for standard and smaller measured tables. Changing
the camera pose or table requires recalibration. Pixels from an unrectified image
or another camera must not be used with the stereo profile, even if image sizes
match. Relative or normalized depth must first be converted to physically
validated metric measurements by the upstream depth provider.

## Ball spacing before table alignment

Use `camera_ball_separation` with two records from the same calibrated camera
frame. Each needs `pixel`, `distance_m`, `distance_kind`, and
`measurement_reference`, using the conventions above:

```python
from billiards_table import camera_ball_separation

spacing = camera_ball_separation(
    first_observation, second_observation, camera_matrix,
    ball_radius_cm=measured_radius_cm)
```

The result contains `center_distance_cm`, `surface_gap_cm`, and both
`camera_centers_m`. The gap is signed; a negative value indicates overlapping
measured spheres. These are nominal measurements, not an accuracy guarantee.
For equal balls resting on a flat table, their 3D center separation equals their
planar separation. Simply subtracting their camera distances does not give that
separation. This function cannot establish table axes or pocket locations.

Optional `camera_id`, `frame_id`, and `calibration_signature` must occur in both
records and match if supplied. Measurements from different cameras must first be
transformed into a common frame using their calibrated relative pose.

## Scalar measurement interface

`TableCalibration.localize_ball_measurement` accepts a pixel plus one measured
distance. Both the distance convention and surface/center reference are required.
`map_camera_measurements` processes a same-frame list using that method:

```python
from billiards_map import map_camera_measurements, planner_state_from_map
from billiards import plan_shots

# calibration is the validated TableCalibration for this exact pixel frame.
# measured_u, measured_v, and measured_z_m come from the measurement provider.
records = [{
    "key": "track-17",
    "frame_id": 42,
    "pixel": {"x": measured_u, "y": measured_v},
    "distance_m": measured_z_m,
    "distance_kind": "camera_z",       # Or "range" for straight-line range.
    "measurement_reference": "surface",  # Or "center" if already corrected.
    "distance_uncertainty_m": 0.003,    # Example tolerance; supply measured error.
    "pixel_uncertainty_px": 1.0,
    "confidence": 0.9,
}]
# Include EVERY remaining ball in records before confirming a real layout.
# The signature comes from the measurement stream's matching calibration;
# do not copy an unrelated profile signature just to satisfy validation.
world_map = map_camera_measurements(
    calibration, records, calibration_signature=measurement_signature,
    labels=reviewed_labels, confirmed=operator_confirmed)
state = planner_state_from_map(world_map, group="solids")
result = plan_shots(state)
```

Optional `radius_px` enables an additional image-size consistency check. Optional
`frame_id` values must be present and equal throughout a batch if any record
includes one. This scalar interface expects a stationary capture; the live
`PoolSession` additionally enforces freshness, accepted captures, and motion
invalidation. Historical maps can be exported for replay, but are not live
observations.

Maps carry `status` (`review`, `confirmed`, or `invalid`), `complete`, `issues`,
and `unmapped`. Only complete, confirmed maps are usable by the public planner
adapter. Balls carry separate `measurement` provenance, including the source pixel,
camera XYZ in meters, distance convention, and surface/center reference. These
measurements are not corrected or projected a second time during planning.

`ball_separations` lists each mapped pair's keys, ball IDs, center distance,
signed surface gap, and combined position margin in centimeters. It includes
unidentified mapped balls during review; unknown identities never remove an
obstacle. Pair distances are calculated from the same table coordinates used by
the planner and overhead display.

## Verify the map

Press **M** after a live scan to compare the measured overhead positions with the
actual table. Check several known points across the table, including off-axis
balls and balls near pocket openings. Confirm that distances between mapped ball
centers agree with physical measurements, and that moving a ball clears old advice.
An incorrect camera-to-table pose can produce a plausible-looking but inaccurate
map; the software's geometry tests do not establish hardware accuracy.
