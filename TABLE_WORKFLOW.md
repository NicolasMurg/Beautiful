# Standalone table alignment and physical-state scripts

These three scripts turn a calibrated camera observation into table coordinates:

```text
calibrate_table.py -> table.json
                         |
capture_ball_measurements.py -> measurements.json + measurement-review.png
                         |
map_table_state.py -> state.json + optional overhead PNG + optional planner input
```

The table map contains ball centers, pairwise distances and surface gaps, cushion
segments, pocket openings, measurement uncertainty, and any unresolved errors.
It describes the current physical layout, without tracking pocketed balls,
turns, or player ownership. Solids and stripes are filters for later shot requests.

Run the commands from the `Beautiful` directory in the existing camera Python
environment. Live capture needs NumPy and GUI-capable OpenCV; optional YOLO
weights use the existing Ultralytics environment. The headless JSON alignment
and mapping paths need NumPy; requesting a PNG also needs OpenCV.

## 1. Align the table once per setup

Camera calibration comes first. Use the existing [stereo calibration
workflow](STEREO_DEPTH.md) to obtain `stereo_calibration.npz` for the current
fixed camera arrangement and resolution. Table alignment does not replace this.

Measure the playing surface **between the inner cushion noses**, the ball
diameter, pocket mouth width, and the camera separation. Place **exactly four
balls**, one near each corner, with a **3-inch (7.62 cm) clear gap between each
ball's surface and each of its two adjacent inner cushion noses**. Remove all
other balls from the camera view during alignment. Use four balls with the same
measured diameter.

The three inches is a surface gap, not a center measurement. The script uses a
center inset of `7.62 cm + ball radius` along both table axes. For a 5.715 cm
diameter ball, that center inset is 10.4775 cm.

```text
          measured length / table x
TL -------------------------------- TR
|    o                          o    |
|                                    | measured width / table y
|    o                          o    |
BL -------------------------------- BR

o is a reference ball, with 3 inches of clear space to each adjacent cushion.
The script labels the balls TL, TR, BR, BL in the camera view.
```

Keep the long table direction left-to-right in the image, with the other axis
running toward the bottom of the image. Both cameras must see the four balls and
the complete playing surface. An elevated side view is supported; a low angle
may hide balls or make depth too inaccurate. You do not enter the camera's
height or setback from the table edge: alignment estimates the rigid table pose.

**The numbers below are examples. Replace every physical measurement with your
actual setup, including when using a smaller table.**

```bash
python3 calibrate_table.py \
  --left-camera /dev/video0 --right-camera /dev/video2 \
  --stereo-calibration stereo_calibration.npz --baseline-inches 5.0 \
  --table-length-cm 254 --table-width-cm 127 \
  --ball-diameter-cm 5.715 --pocket-width-cm 11.5
```

1. Keep the cameras, table, and balls still. Press **Space** to freeze one
   accepted rectified image and its matching depth map. The script automatically
   detects the four reference balls in that frame.
2. Check that the displayed **TL, TR, BR, BL** labels identify the correct four
   balls. No clicks are needed. Fewer or more than four detected balls prevents
   saving; adjust the view or detector settings and press **R** to recapture.
3. Press **Enter** to fit, validate, and save `.local/billiards/table.json`.
   **R** returns to live capture; **Q/Esc** cancels.

Ball references are the default (`--reference-type balls`). Optional
`--model path/to/best.pt` uses your YOLO weights. Adjust
`--min-ball-radius-px` and `--max-ball-radius-px` to the balls' observed image
sizes if necessary. Ball numbers and solids/stripes do not matter for alignment.

Inconsistent scale, reference geometry, or depth is rejected. The default maximum
dimension and fit error is 1 cm; `--max-error-cm` sets the acceptance tolerance,
not a guarantee of accuracy. Correct bad measurements rather than increasing it
to make bad calibration pass. Verify several independent points across the
table before relying on shot directions.

**Remove the four reference balls after saving, then arrange the balls for
play. The table stays aligned.** The saved profile stores the table's position
and orientation; capture and mapping reuse that profile without finding the
reference balls again. The fit accounts for the ball radius and center height
above the felt, so the saved table plane is the felt rather than the elevated
plane through the four ball centers.

Re-run alignment whenever the table or camera rig moves. If the cameras move
relative to one another, redo stereo calibration first. The profile is also
compatible with `live_billiards.py --table-profile ...`.

Pocket recognition is not necessary. The profile uses the measured rectangle's
four corners and two long-rail midpoints as six pocket locations. Measure mouth
widths carefully; optional `--pockets-config pockets.json` accepts the measured
per-pocket overrides described in [BILLIARDS.md](BILLIARDS.md#measured-pockets-and-planner-settings).

## 2. Capture a stationary ball measurement batch

Use the same camera devices, calibration file, and measured baseline:

```bash
python3 capture_ball_measurements.py \
  --left-camera /dev/video0 --right-camera /dev/video2 \
  --stereo-calibration stereo_calibration.npz --baseline-inches 5.0 \
  --table-profile .local/billiards/table.json
```

The script waits for at least one second and three accepted still frames, then
captures a snapshot and exits. It saves:

- `.local/billiards/measurements.json`: pixel coordinates, metric depth,
  distance convention, ball surface reference, identity hints, and errors.
- `.local/billiards/measurement-review.png`: the actual camera image annotated
  with candidate keys. **Keys are not ball numbers.**

Optional `--model path/to/best.pt` adds your existing YOLO weights. `--class-map`
supports a custom class vocabulary; class IDs and color hints are never silently
accepted as ball numbers. The default OpenCV detector works without downloaded
weights but still needs review for missed or false detections. Configure
`--min-ball-radius-px`, `--max-ball-radius-px`, and `--stereo-disparities` as needed
for the actual capture. `--timeout-seconds` defaults to 30.

`--distance-uncertainty-m` adds a measured error allowance to the sampled depth
spread. Default zero means no additional allowance is supplied; it does not mean
the camera is exact. Outputs are a snapshot and have no automatic freshness
guarantee once written. Capture again after any ball moves.

## 3. Generate and review the physical map

Positions can be mapped before ball identities are assigned:

```bash
python3 map_table_state.py \
  --table-profile .local/billiards/table.json \
  --measurements .local/billiards/measurements.json \
  --output .local/billiards/state.json \
  --image .local/billiards/table-map.png
```

Review the overhead map against the real layout and the captured image. Unknown
identities remain visible. A failed depth reading stays in `unmapped` with an
explanation; it does not disappear as an assumed pocketed ball. The map includes:

| Field | Meaning |
| --- | --- |
| `units`, `coordinate_frame` | Centimeters; origin at top-left inner cushion intersection |
| `balls` | Measured centers (`x`, `y`), keys, optional reviewed IDs, uncertainty and provenance |
| `ball_separations` | Each pair's center distance, signed surface gap and combined position margin |
| `table.pockets`, `rails` | Calibrated mouths, blocked flags and cushion segments |
| `status`, `complete`, `issues`, `unmapped` | Whether the map needs review or has unusable measurements |

For shot planning, make a labels JSON file that maps the keys **in this capture**
to actual ball numbers. Use `0` for the cue ball and `1` through `15` for the
numbered balls. For example, only if key `1` is the cue ball and key `2` is ball 3:

```json
{"1": 0, "2": 3}
```

Label every ball currently on the table, using the actual keys and numbers.
Do not confirm a layout missing an occluded ball. For interactive correction of
missed or false detections, use the existing `live_billiards.py` scan/review
interface with the saved table profile; it supports adding a real ball by
clicking its observed pixel. The standalone capture files contain no invented
positions or automatic layout repair.

After the complete snapshot has been reviewed, save the labels as
`.local/billiards/labels.json` and export planner input:

```bash
python3 map_table_state.py \
  --table-profile .local/billiards/table.json \
  --measurements .local/billiards/measurements.json \
  --labels .local/billiards/labels.json --confirmed \
  --output .local/billiards/state.json --image .local/billiards/table-map.png \
  --planner-state .local/billiards/planner-state.json

python3 plan_billiards.py --state .local/billiards/planner-state.json \
  --confirmed --question "best shot for stripes"

python3 plan_billiards.py --state .local/billiards/planner-state.json \
  --confirmed --question "best shot for solids"
```

Check that mapping succeeded before using planner output. Invalid measurements
save a diagnostic map and return exit code 2; no new planner input is written.
An ordinary unconfirmed map returns 0 and remains review-only. `--confirmed`
requires complete identities and geometry; it does not assert any missing ball's
history. All other balls remain obstacles when requesting solids or stripes.

## Using your own metric depth provider

You can skip the capture script when your detector already provides image
positions and metric distances. Scalar distances alone are insufficient. Pixels
must be undistorted/rectified and match the supplied intrinsic matrix; raw pixels
cannot simply be relabeled as rectified. Relative model depth must first be
converted to physically validated metric measurements by the provider.

For headless four-ball alignment, arrange the same four reference balls with
the three-inch surface gaps described above, and save a JSON object with:

| Field | Required value |
| --- | --- |
| `pixel_space` | `"rectified"` |
| `image_size` | `[width_pixels, height_pixels]` |
| `camera_matrix` | Calibrated 3 x 3 intrinsic matrix for those pixels |
| `calibration_signature` | Nonempty identifier/object tied to that exact calibrated camera frame |
| `balls` | Exactly four reference-ball records, in any input order |

Each ball record contains `pixel: {"x": u, "y": v}`, `distance_m`,
`distance_kind`, and `measurement_reference`. Use `"camera_z"` for optical-axis
depth or `"range"` for the straight-line distance from the camera center. Set
`measurement_reference` to `"surface"` for the visible central surface or
`"center"` if the provider has already corrected to the ball center. The script
applies the ball-radius correction only for surface measurements, then accounts
for the centers' height above the felt when fitting the table profile.

The balls are ordered automatically from their pixels; names are not needed.
Optional `radius_px` supplies an image-size consistency check. If you provide
`frame_id` or `camera_id`, they must be present and equal in all four records.

```bash
python3 calibrate_table.py --measurements .local/billiards/reference-balls.json \
  --table-length-cm 254 --table-width-cm 127 \
  --ball-diameter-cm 5.715 --pocket-width-cm 11.5
```

Again replace those example dimensions with your actual measurements. For ball
mapping, use a JSON object containing the same `pixel_space`, `image_size`, and
provider `calibration_signature`, plus a `measurements` list. Optional
`observed_at` is a UNIX timestamp in seconds. Each ball record needs:

| Field | Meaning |
| --- | --- |
| `key` | Unique string for reviewing this candidate |
| `pixel` | `{"x": u, "y": v}`, the projected ball center |
| `distance_m` | Finite positive metric measurement |
| `distance_kind` | `"camera_z"` or `"range"` |
| `measurement_reference` | `"surface"` for the visible central surface, or `"center"` if already corrected |
| `distance_uncertainty_m` | Optional nonnegative measurement error allowance |
| `pixel_uncertainty_px` | Optional positive image localization tolerance; default 1 pixel |
| `radius_px`, `confidence` | Optional image-size check and confidence in [0, 1] |
| `frame_id`, `camera_id` | Optional; if present, must match throughout the batch |

An unusable detection can instead contain its `key`, common frame/camera IDs,
pixel if available, and an `error` string. Keep it in the batch so mapping reports
the missing obstacle rather than ignoring it. Identity hints are not labels;
reviewed labels are supplied separately using `--labels`.

The mapper checks that the measurement signature and resolution match the table
profile. Supply the signature from the actual upstream calibration; do not copy
an unrelated profile's signature to bypass that check. Mixing observations from
different cameras requires first transforming them into a common calibrated
frame. After this contract is satisfied, run `map_table_state.py` exactly as above.

## Optional flat-marker alignment

The older flat-marker workflow remains available with
`--reference-type markers --marker-inset-cm <measured-center-inset>`. Place four
flat markers on the felt with the same measured center inset from both adjacent
cushion lines. After **Space**, click their centers in **TL, TR, BR, BL** order,
then press **Enter** to save. **Backspace** undoes a click. Printable markers are
in [pool_table_markers.svg](pool_table_markers.svg).

Headless marker mode uses a `markers` list ordered TL, TR, BR, BL, with `name`,
`pixel`, `distance_m`, and `distance_kind` for each flat marker. These distances
refer to the marker centers on the felt; no ball-radius correction is applied.
Use the same pixel calibration metadata and matching frame/camera IDs required
for ball alignment. The saved profile persists after removing the markers too.

## Validation

Automated tests cover known 3D table poses, both distance conventions, surface
and center measurements, calibration mismatches, missing measurements, frozen
reference capture, and planner export. They do not establish hardware accuracy or
detect every missed ball. Compare reconstructed positions and ball separations
against independent physical measurements across the actual table.

All default outputs live under ignored `.local/billiards/`. Nothing is uploaded.
