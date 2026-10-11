# Camera-based billiards shot adviser

`live_billiards.py` uses the existing calibrated stereo cameras to observe a real
table and suggest a direct pot from a reviewed, stationary ball layout. The
OpenCV window shows the actual camera image, detected balls, and calculated aim
lines. No browser, generated table layout, cloud service, or language model is
required.

This version advises a shot before you play. It does not track balls through a
high-speed break, simulate spin, or guarantee a pot. Motion invalidates the old
advice so you can scan and review the next layout.

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
5. Select the appropriate group with **G**: solids, stripes, or open table.
6. Press **C** only after every visible ball has the correct identity and all
   absent IDs really are already pocketed. This is the operator's confirmation
   of the table state. If a ball is hidden or off-camera, uncover it or change
   the view before confirming; absence from detection does not establish that
   it was pocketed.
7. Press **A** for the best available direct pot, or **N** for the next
   alternative. Press **?**, type a supported shot question, and press **Enter**.
   You can also enter a plain question in the terminal while the app runs.

The request handler supports a limited billiards command vocabulary, including
requests about a selected ball or group. It is not an open-ended conversational
model. Ball identities come from the reviewed camera observation, not from a
spoken request inventing or moving a ball.

Examples include `What is my best shot?`, `I am stripes, what should I shoot?`,
`Where should I aim for ball 10?`, and `Aim for the red solid`. The request must
agree with the confirmed layout and legal target group. A color alone is
ambiguous on an open table because both a solid and a stripe share that color.

### Controls

| Control | Action |
| --- | --- |
| **T** | Recalibrate the table reference |
| **Space** | Scan a stationary physical layout |
| Click, number, **Enter** | Confirm the selected ball's identity |
| **Shift-click** | Add a missed ball at a real camera pixel with valid depth |
| Right-click | Reject a false candidate |
| **Delete** | Clear the selected ball's label |
| **C** | Confirm the complete reviewed table state |
| **G** | Cycle solids, stripes, and open-table selection |
| **A** | Request the best direct-pot candidate |
| **N** | Show the next alternative |
| **?**, text, **Enter** | Ask using the camera window |
| **P** | Record a spoken question when started with `--voice` |
| **Esc** | Cancel text entry; quit when not entering text |
| **Q** | Quit outside text entry |

Detected motion, a rejected camera pair, invalid depth, or stale observations
disable advice. Once the scene is usable and still again, scan and confirm the
new layout. This deliberate review is especially important after any shot;
previous ball labels are not trusted through collisions or occlusion.

## What the overlay means

The adviser ranks legal direct-pot candidates from the confirmed state using
geometric difficulty and path clearance. The displayed best candidate is the
best under that scoring model, not a guaranteed optimal move in a full game.
The overlay identifies the target ball and pocket and draws:

- The ideal ghost-ball center: where the cue ball's center would be at contact
  for an ideal collision sending the object ball toward that pocket.
- The cue ball's path toward the ghost-ball center and the object ball's path
  toward the pocket.
- The **aim angle**, measured clockwise from table-right in the chosen table
  coordinate system, separately from the **cut angle**, which describes the
  change between the incoming cue-ball direction and outgoing object-ball
  direction. These are not image-space angles in an oblique camera view.
- An extension of the aim line to a cushion line, reported with a diamond
  coordinate: 0–8 along a long side from the left, and 0–4 along a short side
  from the top, using the chosen table orientation.

The cushion-line intersection is a reference for aiming. It is not a bank-shot
instruction, the predicted cushion contact of the cue ball, or necessarily the
position of a physical diamond embedded in the outside rail. The model does not
account for spin, throw, cloth speed, cushion rebounds, or the cue ball's full
motion after contact.

## Detection and optional YOLO

The default detector combines OpenCV circle and color proposals inside the
table polygon. It can operate without downloaded pool-ball weights. Radius
bounds and good framing matter: adjacent balls, reflections, shadows, and small
ball images can cause missed or extra candidates. Review the result before
asking for a shot.

Add `--model yolo26n.pt` to supplement these proposals using existing local
Ultralytics weights. Generic COCO weights know the `sports ball` class; they do
not recognize pool-ball numbers. Standard color/stripe heuristics can suggest
numbers but never confirm them. The operator's labels establish identity.

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
5. Test the selected group and an eight-ball-only situation against the actual
   game state. Continue to apply the rules used for your match.
6. Attempt repeated, simple shots and record observed results separately from
   the geometry checks. Keep the rig unchanged between comparisons.

For a portable demo, physical miniature balls and a measured miniature table
can use this same camera application after calibration. Enter their actual ball
diameter, table dimensions, and pocket opening. Miniature shot dynamics are not
a validation of the full-size table's dynamics.

## Geometry reference

The stereo coordinates follow the rectification and triangulation conventions
in [OpenCV's calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
For the ideal collision geometry behind ghost-ball aiming, see
[David Alciatore's pool physics article](https://drdavepoolinfo.com/physics_articles/Alciatore_pool_physics_article.pdf).
Target-group and eight-ball eligibility use a limited subset of the
[WPA rules](https://wpapool.com/wp-content/uploads/2026/01/2026.01.02-WPA-Rules.pdf).
The app does not adjudicate a full match, called-shot compliance, break rules,
fouls, or scratches; use the rules agreed for your game.
