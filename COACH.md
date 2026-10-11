# AI pool coach through the JBL speaker

`live_billiards_coach.py` runs the existing physical billiards adviser
([BILLIARDS.md](BILLIARDS.md)) with a gentle spoken coach:

> "Take a deep breath and line up from diamond 3 on the left rail toward diamond
> 5.5 on the top rail. A quarter-ball hit sends the 1 into hole 2, the top middle
> side pocket."

The existing planner still decides every shot. A local Hugging Face model
(`Qwen/Qwen2.5-1.5B-Instruct`) only puts the planner's numbers into friendly
words and understands casual questions. Every model reply is checked: if it
contains a number the planner did not produce, or leaves out the ball, hole or
diamonds, the coach speaks a fixed template instead. Nothing is sent to a cloud
service. The existing billiards files are unchanged; `live_billiards.py` still
works exactly as before.

## What the coach says

- **Line up from diamond X**: the aim line extended *behind* the cue ball to a
  rail. Stand there and sight through the cue ball.
- **Aim toward diamond Y**: the same line extended forward past the ghost ball
  (the planner's existing rail reference). Neither is a bank shot.
- **Hole 1–6**: numbered clockwise from top left in the calibrated table view:
  1 top left, 2 top middle, 3 top right, 4 bottom right, 5 bottom middle,
  6 bottom left. They are drawn as H1–H6 just outside each pocket on the camera view.
- Diamonds count 0–8 along the long rails from the left and 0–4 along the short
  rails from the top, rounded to the nearest half diamond.
- **Hit**: straight-in, three-quarter, half, quarter ball, or thin cut, from the
  planner's cut angle.

Spin, speed and the cue ball's path after contact are still not predicted.

## Install once on the Jetson

In the Python environment that already runs YOLO (do not reinstall torch):

```bash
python3 -m pip install transformers huggingface_hub
bash tools/setup_coach.sh
```

This downloads the coach model (~3 GB) into the Hugging Face cache and the Piper
voice runtime into `.local/coach/` (ignored by Git). Internet is needed only for
this step. For a lighter model: `bash tools/setup_coach.sh Qwen/Qwen2.5-0.5B-Instruct`
and add `--coach-model Qwen/Qwen2.5-0.5B-Instruct` when running.

## Pair the JBL speaker (Bluetooth)

Put the JBL in pairing mode (hold the Bluetooth button until it flashes), then:

```bash
bluetoothctl
  power on
  scan on            # wait for "JBL ..." and note its MAC, e.g. AA:BB:CC:DD:EE:FF
  pair AA:BB:CC:DD:EE:FF
  trust AA:BB:CC:DD:EE:FF
  connect AA:BB:CC:DD:EE:FF
  quit
pactl list short sinks                  # find bluez_output.AA_BB_CC_DD_EE_FF...
pactl set-default-sink bluez_output.AA_BB_CC_DD_EE_FF.1   # use the exact name shown
python3 coach.py --say "Hello from your pool coach"
```

You can also pick the JBL in Ubuntu's Settings → Sound → Output. Run the coach as
your normal desktop user, not with `sudo`, or it cannot reach the audio server.
`trust` lets the JBL reconnect automatically next time. Each spoken line starts
with 0.4 s of silence so a speaker waking from idle does not cut the first word.
An AUX cable or USB connection works the same way once it is the default output.

## Run it

```bash
python3 coach.py --demo                 # sample layout, no cameras: prints + speaks
python3 live_billiards_coach.py \
  --left-camera /dev/video0 --right-camera /dev/video2 \
  --stereo-calibration stereo_calibration.npz --baseline-inches 5.0 --voice
```

Use the same camera arguments as `live_billiards.py`. Then scan, label each ball,
press **C** to confirm, and press **A**. The JBL speaks the coached shot and the
line also appears at the bottom of the side panel. **N** coaches the next option.

Ask by typing in the terminal, pressing **?**, or pressing **P** with `--voice`:

- `what's my best shot?`, `aim for ball 10` (handled directly, no model needed)
- `which one's easiest, I've got stripes` (casual: the model maps it to a supported request)
- `why that one?` (explains the planner's reasons)
- `give me another option` (next alternative)

A casual question pauses the camera view while the model reads it, typically a
few seconds. Replies about a ball are still checked against the confirmed
layout; the model cannot add or move balls. Off-topic questions get a short hint.

| Option | Meaning |
| --- | --- |
| `--coach-model` | Hugging Face model id; `none` uses templates only |
| `--coach-device` | `0` (GPU) or `cpu` |
| `--coach-timeout` | Seconds allowed per model reply (default 8) |
| `--tts` | `auto` (Piper, then espeak), `piper`, or `espeak` |
| `--piper-voice` | Another Piper `.onnx` voice from `rhasspy/piper-voices` |

If the model or voice cannot load, the program says so and continues with the
template wording or on-screen text. It never silently changes the shot.

## Verified so far

Automated tests (`tests/test_coach.py`) use fake models and voices. On a Windows
laptop CPU, `coach.py --demo` produced validated coached lines with
`Qwen2.5-1.5B-Instruct` (~6–7 s per line on CPU) and `Qwen2.5-0.5B-Instruct`
(~2 s, less reliable wording). Not yet measured: Jetson GPU latency, memory use
alongside stereo depth, Piper on the Jetson, and the JBL connection.
