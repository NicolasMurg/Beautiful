# Camera-based billiards shot adviser

`live_billiards.py` uses the existing calibrated stereo cameras to observe a real
table and rank direct pots and one-cushion object-ball banks from a reviewed,
stationary ball layout. The
OpenCV window shows the camera image or a reconstructed overhead metric map,
detected balls, and calculated aim lines. No browser, invented ball layout, cloud
service, or language model is required.

This version advises a shot before you play. It does not track balls through a
high-speed break, simulate arbitrary spin, or guarantee a pot. Motion invalidates the old
advice so you can scan and review the next layout.

## Measurements become a virtual table before planning

For separate alignment, measurement capture, and map-export commands, see
[the standalone table workflow](TABLE_WORKFLOW.md). Its saved profile also works
with this live adviser.

The processing stages are explicit:

```text
Object image position + calibrated camera distance
    -> 3D position in the calibrated camera frame
    -> physical x/y coordinates on the table in centimeters
    -> virtual map of balls, pocket mouths, and cushion segments
    -> reviewed map -> candidates for the requested balls -> ranked advice
```

`billiards_table.py` performs the coordinate conversion. `billiards_map.py`
builds the world map, preserves measurement provenance and uncertainty, and
provides the planner's state. `billiards_map_view.py` draws that same map from
above with a uniform physical scale. The planner does not use camera distance
as if it were a ball's position on the felt.

Press **M** to toggle camera/map view, or start with `--map-view`. The map shows
the table dimensions, centimeter grid, ball identities, position uncertainty,
pocket openings and blocked flags, and the selected shot paths after confirmation.
Switch back to the camera to label detections or mark a missed real ball. No map
control lets you invent or drag a ball to a preferred position. Motion, rejected
capture, or unusable depth invalidates both the map's guidance and the shot.

Distance alone is insufficient: two balls equally far from a camera can be far
apart from each other. Each ball also needs its image position and matching
camera intrinsics. These give 3D positions in the camera frame and therefore
ball-to-ball spacing, without needing the camera's height or distance behind the
table edge. A camera mounted to the side can work; visibility and depth accuracy
still need checking across the whole table.

Placing those balls relative to cushions and pockets additionally needs the
camera-to-table pose. The four-marker alignment below estimates it; you do not
need to measure the camera's setback from the table edge. Pocket recognition is
not required: measured table dimensions, corner alignment, and pocket openings
define the six-pocket geometry. Without either measured or reliably detected
pocket locations, the system cannot recommend a pot.

Current stereo measurements are optical-axis Z in
meters, sampled at the visible ball surface. An explicit scalar-measurement API
also supports straight-line range and center-distance measurements without
confusing those conventions. See [BILLIARDS_MAP.md](BILLIARDS_MAP.md) for the
input contract and coordinate math.

## What to prepare

- The Jetson environment that already runs `live_yolo_stereo.py`, including
  NumPy and a GUI-enabled OpenCV build. Use a desktop display for the camera
  window; a headless OpenCV package cannot display it.
- Two rigidly mounted webcams and the existing `stereo_calibration.npz` for
  their current arrangement and image resolution.
- The measured camera separation, playing-surface length and width, ball
  diameter, and smallest pocket opening you want the planner to use.
- Four flat, matte, textured markers. Print
  [pool_table_markers.svg](pool_table_markers.svg) at 100% size. Its patterned
  squares are 35 mm wide; the labels sit outside the sample patterns. These
  provide stereo texture and do not replace camera calibration.

Frame the entire playing surface with both cameras. Balls must be large enough
in the images to distinguish individual candidates; inspect the live view at
its actual capture resolution. A higher mounting position changes that tradeoff.
Changing the cameras' spacing, relative angle, lens settings, or capture
resolution requires checking or repeating stereo calibration using the existing
[stereo workflow](STEREO_DEPTH.md).

Detection alone is insufficient: each ball also needs usable stereo samples.
The localizer rejects ball images with radii below 3 pixels and rejects sparse
or inconsistent depth patches. A ball exceeding that cutoff is not automatically
large enough for precise aiming. Check balls across the full table, including
white and glossy surfaces and the edges of the stereo overlap.

The existing matcher defaults to a 192-pixel disparity search, which can leave
a substantial invalid region near an image edge. Keep your validated stereo
setting when it differs, using `--stereo-disparities`. A smaller search such as
64 is a possible configuration to test, not a universal fix: it must still cover
the disparities at the actual working distances. Supported search widths are
positive multiples of 16 and smaller than the image width.

## Align the table with the stereo measurements

Measure the playing surface between the cushion noses, excluding the outer
wooden rails. Choose a table coordinate convention with **TL to TR along the
measured length** and **TL to BL along the measured width**:

```text
TL ----------------------- TR
 |                          |
 |                          |  width
 |                          |
BL ----------------------- BR
             length
```

Place each marker's center the same measured inset from its two adjacent
cushion lines. For example, `--marker-inset-cm 10` means the TL marker center is
10 cm inward from each of the TL corner's two cushion lines. Keep the patches
flat on the felt. Measure their center positions; the printed square's edge is
not the reference point.

The following values are examples, not measurements of your rig or table.
Replace the camera devices and **every physical measurement** with your own:

```bash
python3 live_billiards.py \
  --left-camera /dev/video0 \
  --right-camera /dev/video2 \
  --stereo-calibration stereo_calibration.npz \
  --baseline-inches 5.0 \
  --table-length-cm 254 \
  --table-width-cm 127 \
  --ball-diameter-cm 5.715 \
  --pocket-width-cm 11.5 \
  --marker-inset-cm 10
```

Click the marker centers in this order: **TL, TR, BR, BL**. The program samples
the raw stereo depth at those locations and fits the known physical table to
the measured 3D points. Invalid or inconsistent measurements must be corrected;
there is no fallback that silently substitutes an image-only homography.

The local table profile is saved in `.local/billiards/table.json`. The existing
`.gitignore` rule for `.local/` keeps it out of normal Git commits. Keep
`PROJECT_CONTEXT.txt` ignored as before.

With the same cameras and unchanged table setup, subsequent runs can reuse the
profile:

```bash
python3 live_billiards.py \
  --left-camera /dev/video0 \
  --right-camera /dev/video2 \
  --stereo-calibration stereo_calibration.npz \
  --baseline-inches 5.0
```

Use your actual baseline again. Press **T** to align the table again after the
rig or table moves; this uses the current profile's dimensions and marker inset.
Supplying the three options `--table-length-cm`, `--table-width-cm`, and
`--pocket-width-cm` together starts a new table calibration even if a profile
already exists. To change physical dimensions, ball size, or marker inset,
provide those three measurements plus the updated options and recalibrate.
If the cameras move relative to one another, redo stereo calibration first.

## Review the real layout and ask for a shot

1. Let all balls stop and keep hands and cues out of the playing surface. After
   the view has been still for at least a second, press **Space** to scan it.
2. Review the frozen ball candidates. Camera capture continues behind the
   review so the program can detect a changed physical layout.
3. Click a candidate, type its number, and press **Enter**. Use **0 for the cue
   ball** and **1–15 for numbered balls**. Color suggestions are only hints.
4. **Shift-click** a missed ball's center to add a candidate from a real pixel
   with observed stereo depth. Right-click a false candidate to reject it.
   **Delete** clears the selected candidate's label.
5. Optionally choose a default target filter with **G**: solids, stripes, or
   open (all ordinary numbered balls). You can specify a group in each request.
6. Press **C** only after every ball currently on the table is represented with
   the correct identity. If a ball is hidden or off-camera, uncover it or change
   the view before confirming so it is included as a possible obstruction.
   Confirmation records a complete physical snapshot, not any ball's history.
7. Press **A** for the best available modeled shot, or **N** for the next
   alternative. Press **?**, type a supported shot question, and press **Enter**.
   You can also enter a plain question in the terminal while the app runs.

The request handler supports a limited billiards command vocabulary, including
requests about a selected ball or group. It is not an open-ended conversational
model. Ball identities come from the reviewed camera observation, not from a
spoken request inventing or moving a ball.

Examples include `What is my best shot?`, `Best shot for stripes`,
`Best shot for solids`, `Where should I aim for ball 10?`, and `Aim for the red
solid`. Each group request filters targets in the same current layout; it does
not assign a group to a player or change the default filter. Balls outside the
requested group remain obstructions. A color alone is ambiguous with the open
filter because both a solid and a stripe share that color.

The adviser does not track pocketed balls, turns, or player ownership. If no balls
from the requested group are present, it returns `no_shot`; it never automatically
switches to the eight ball. An explicit request for ball 8 evaluates its geometry
without deciding whether playing it is allowed under the match rules.

Additional typed and spoken commands include `Best bank shot`, `Best direct shot`,
`Next shot`, `Why this shot?`, `Repeat shot`, `Direct shots only`, and
`Allow bank shots`. `Block the top middle pocket` (or `Block TM`) excludes a
temporarily unusable pocket; `Unblock TM` restores it. These pocket flags persist
through rescans within the current session. Ordinary ball obstructions are
calculated from their measured positions without needing a block command.
Commands never override the fresh-frame and reviewed-layout requirements.

### Controls

| Control | Action |
| --- | --- |
| **T** | Recalibrate the table reference |
| **Space** | Scan a stationary physical layout |
| Click, number, **Enter** | Confirm the selected ball's identity |
| **Shift-click** | Add a missed ball at a real camera pixel with valid depth |
| Right-click | Reject a false candidate |
| **Delete** | Clear the selected ball's label |
| **L** | Copy unique custom-model ball labels, then review them before C |
| **C** | Confirm the complete reviewed table state |
| **G** | Cycle the default target filter: solids, stripes, or open |
| **A** | Request the best modeled direct or bank candidate |
| **N** | Show the next alternative |
| **M** | Toggle live camera and the measured overhead table map |
| **?**, text, **Enter** | Ask using the camera window |
| **P** | Record a spoken question when started with `--voice` |
| **Esc** | Cancel text entry; quit when not entering text |
| **Q** | Quit outside text entry |

Detected motion, a rejected camera pair, invalid depth, or stale observations
disable advice. Once the scene is usable and still again, scan and confirm the
new layout. This deliberate review is especially important after any shot;
previous ball labels are not trusted through collisions or occlusion.

## What the overlay means

The adviser ranks direct pots and one-cushion object-ball banks for the requested
targets from the confirmed state. It checks cue and object paths against all observed balls,
their supplied position uncertainty, cushion segments, and pocket jaws. A bank
must rebound from a real cushion segment, away from pocket openings. The displayed
best candidate is the best under this finite search and scoring model, not a
guaranteed optimal move in a full game.
The overlay identifies the target ball and pocket and draws:

- The ideal ghost-ball center: where the cue ball's center would be at contact
  for an ideal collision sending the object ball toward that pocket.
- The cue ball's path toward the ghost-ball center and every leg of the object
  ball's path toward the pocket, including the rebound point for a bank.
- Purple dashed cue-ball continuation under the conditional stun model.
- The **aim angle**, measured clockwise from table-right in the chosen table
  coordinate system, separately from the **cut angle**, which describes the
  change between the incoming cue-ball direction and outgoing object-ball
  direction. These are not image-space angles in an oblique camera view.
- An extension of the aim line to a cushion line, reported with a diamond
  coordinate: 0–8 along a long side from the left, and 0–4 along a short side
  from the top, using the chosen table orientation.

The dashed cushion-line intersection is a reference for aiming, separate from
the object ball's bank point. It is not necessarily the position of a physical
diamond embedded in the outside rail.

Ranking considers cut angle, cue/object travel, remaining clearance, pocket
margin, bank difficulty, detection quality, position uncertainty, required pace,
and a conditional scratch estimate. JSON output includes the score penalties,
reasons, and counts of rejected candidates. Scores are not calibrated make odds.

The scratch estimate assumes an ideal **stun impact** and a simple cloth-drag
model. It traces cue-ball motion until a stop, scratch, another ball, uncertain
pocket jaw, or a bounded number of cushion contacts. Unknown outcomes are marked
unknown and penalized. A low estimate does not establish that a follow/draw stroke
is safe. Soft/medium/firm pace and numerical speed are model estimates that need
physical calibration. The search does not yet include combinations, cue-ball
kicks, jumps, spin/throw, defensive shots, or planning multiple shots ahead. When
no modeled route is available, it returns `no_shot` with rejection diagnostics.

Balls hanging inside a pocket mouth remain targets and blockers when their
centers are still inside the measured table rectangle and clear the actual
cushion segments. Centers on the unmeasured shelf beyond that rectangle require
operator review; the system does not invent shelf geometry.

### Measured pockets and planner settings

Default pockets are the four corners and two long-rail midpoints. To supply
individual opening widths, a shifted side-pocket location, or a blocked pocket,
save a JSON list under `.local/billiards/pockets.json` and add
`--pockets-config .local/billiards/pockets.json`:

```json
[
  {"id": "TM", "x": 126.5, "width": 12.0},
  {"id": "BR", "width": 11.0, "blocked": true}
]
```

Those are example measurements for a 254 cm table; replace them with your own.
All coordinates and lengths are centimetres. IDs are `TL`, `TM`, `TR`, `BL`, `BM`,
`BR`. Unspecified fields use the table's defaults. Corner centers remain at the
cushion intersections; `TM` and `BM` may move along their respective long rails.
Overlapping openings, off-rail centers, and openings narrower than a ball are
rejected. Configured pockets are saved with the table profile and used on later
runs; an empty list restores the standard geometry. YOLO pocket boxes remain
visual review hints and never silently replace measured geometry.

Local ball uncertainty combines calibration residuals, depth spread, and a
one-pixel localization tolerance. This is an engineering margin, not a statistical
confidence interval or complete model of systematic camera error. Use
`--position-margin-cm 0.2` to add an additional measured error allowance, and
`--direct-only` to start without bank candidates.

Optional `--planner-config .local/billiards/planner.json` accepts:

```json
{
  "allow_banks": true,
  "max_cut_angle_deg": 80,
  "rolling_deceleration_cm_s2": 20,
  "cushion_restitution": 0.8,
  "pocket_entry_speed_cm_s": 22.86,
  "position_uncertainty_cm": 0.2
}
```

The drag, restitution, and entry speed above are example assumptions, not measured
properties of your table. Calibrate them from actual rolling/rebound observations
before relying on numerical speed or stopping distance. Changing table/ball size
requires new geometry and checking these dynamics as well. Command-line margin
and direct-only options override the corresponding JSON settings.

## Detection and optional YOLO

The default detector combines OpenCV circle and color proposals inside the
table polygon. It can operate without downloaded pool-ball weights. Radius
bounds and good framing matter: adjacent balls, reflections, shadows, and small
ball images can cause missed or extra candidates. Review the result before
asking for a shot.

Add `--model path/to/best.pt` to supplement these proposals using existing local
Ultralytics weights. Generic COCO weights know the `sports ball` class; they do
not recognize pool-ball numbers. Custom labels such as `cue_ball`, `ball_1`, and
`ball_15` can supply number suggestions. For a different model vocabulary, pass
`--class-map .local/billiards/classes.json`, a JSON object such as
`{"white_cue": 0, "nine_ball": 9, "17": 8}`. Numeric keys are model class
indices; their values are actual ball IDs, not inferred COCO IDs.

Use `--yolo-confidence 0.5` to change the detection threshold. Overlapping boxes
with contradictory identities stay unconfirmed. **L** copies unique model label
suggestions into the review; it skips conflicts, repeated numbers, and candidates
without valid depth. Appearance-only color guesses are never bulk-accepted.
Review the copied labels and all missing balls before **C**. Model detections of
pockets, rails, table, and cue sticks appear as unconfirmed review hints when the
weights contain those classes. This code does not train or supply custom weights.

Use `python3 live_billiards.py --help` for the available detector and capture
options. Keep using the Jetson's working GPU environment rather than replacing
its PyTorch or CUDA packages to enable this app.

## Optional voice input and spoken replies

Follow [SPEECH.md](SPEECH.md) once to build the local Whisper runtime and download
its model. Add `--voice` to the camera command. Press **P**, wait until the
status says recording, and then ask for a shot. Microphone selection is automatic.
Whisper transcribes the utterance locally; the same limited request handler
processes typed and spoken input.

For optional local spoken replies, install a speech synthesizer on the Jetson:

```bash
sudo apt install espeak-ng
```

Then add `--speak` to the camera command. An existing `espeak` executable is also
supported. Voice recognition and speech output are separate options; neither is
required to use the camera window and keyboard controls. Setup downloads require
internet, but recognition, planning, and speech output run locally afterward.

## Validate on the physical setup

The automated tests check synthetic geometry, invalid measurements, candidate
detection, identity handling, and request behavior. They do not establish camera
accuracy, Jetson frame rate, or successful physical shots. This implementation
has not been validated with the project's actual cameras or table.

For a first table session:

1. Check measured table positions against several independent reference points,
   including points away from the four fitting markers.
2. Arrange an easy unobstructed pot. Confirm all ball labels and verify the
   target, pocket, ghost-ball point, and displayed direction by inspection.
3. Add a blocking ball. Check that the obstructed path is rejected or another
   direct pot is offered, including a no-shot result when appropriate.
4. Move a ball after confirmation and check that the old recommendation clears.
   Also check a missed ball, a false detection, and unavailable stereo depth.
5. Ask for solids and then stripes from the same snapshot. Check that each
   request selects the intended targets while all other balls remain obstacles.
6. Attempt repeated, simple shots and record observed results separately from
   the geometry checks. Keep the rig unchanged between comparisons.

Also test an unobstructed bank, a ball blocking each bank leg, a pocket marked
blocked, a narrow pocket, and an intended bank point over a pocket opening. Compare
scratch warnings with real cue-ball behavior at several speeds. Repeat with the
table's measured uncertainty margin rather than judging nominal lines alone.

For a portable demo, physical miniature balls and a measured miniature table
can use this same camera application after calibration. Enter their actual ball
diameter, table dimensions, and pocket opening. Miniature shot dynamics are not
a validation of the full-size table's dynamics.

## Geometry reference

The stereo coordinates follow the rectification and triangulation conventions
in [OpenCV's calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
For the ideal collision geometry behind ghost-ball aiming, see
[David Alciatore's pool physics article](https://drdavepoolinfo.com/physics_articles/Alciatore_pool_physics_article.pdf).
The bank model uses the [mirror aiming construction](https://drdavepoolinfo.com/faq/bank-kick/mirror/);
real speed and spin require adjustments. Conditional cue continuation follows
the [stun tangent-line model](https://drdavepoolinfo.com/faq/stun/shot/).
Target groups are request filters only. The app does not adjudicate a match,
assign player groups, infer pocketing history, or decide eight-ball eligibility;
the player remains responsible for the rules used for the game.

## Export and replay observed layouts

Add `--export-state .local/billiards/latest.json` to the live command. Every **C**
confirmation atomically saves the reviewed geometry, ball IDs, uncertainty,
confidence, current planner settings, and a `virtual_map` snapshot with camera
measurement provenance. This is a historical snapshot; it is
never automatically trusted as a current camera observation.

Run the same planner without cameras, OpenCV, or model packages:

```bash
python3 plan_billiards.py --state .local/billiards/latest.json --confirmed
python3 plan_billiards.py --state .local/billiards/latest.json --confirmed --question "best bank shot"
python3 plan_billiards.py --state .local/billiards/latest.json --confirmed --question "best shot for stripes"
python3 plan_billiards.py --state .local/billiards/latest.json --confirmed --question "best shot for solids"
```

`--confirmed` declares that every ball currently on the table is represented in
the reviewed snapshot. Omitted numbers have no inferred status. The command emits
strict JSON, including ranked shots
and diagnostics, and rejects invalid input. It never substitutes an example layout.
Application code can call `billiards.plan_shots(state, question)` directly; the
required state is `table` and `balls`, with an optional `group` default target
filter (default `open`) and optional `planner` settings.
Each ball contains `id`, `x`, `y`, and optional `confidence` and
`position_uncertainty_cm`. The command's `--help` shows the schema.

For development, run `python -m pytest -q` with NumPy, OpenCV, and pytest available.
Tests cover geometry, blockers, pockets, bank legs, conditional scratch paths,
invalid data, model-label conflicts, stale observations, spoken-command routing,
JSON replay, and camera-to-overlay integration with synthetic stereo observations.
They do not replace validation on the actual rig.
