# Local speech recognition on Jetson

Whisper speech-to-text uses the local `whisper.cpp` runtime and the pretrained
English `tiny.en` model. This baseline runs on the CPU with two threads, keeping
GPU inference available to YOLO/depth. No Python speech package, cloud account,
API key, or replacement of PyTorch/CUDA is needed.

## Install once on the Jetson

Open a terminal in `Beautiful`. Install build tools and ALSA microphone capture:

```bash
sudo apt install git cmake build-essential curl alsa-utils
bash tools/setup_whisper.sh
```

The setup pins `whisper.cpp` to `v1.7.6`, builds its CPU CLI, and downloads
`ggml-tiny.en.bin`. It does not change JetPack or GPU libraries. Internet is needed
for this setup; recognition uses local files and works offline afterward. The
runtime in `.local/` and weights in `models/whisper/` are ignored by Git.

## Test the microphone and transcription

```bash
python3 voice_to_text.py --list-devices
python3 select_microphone.py
python3 voice_to_text.py
```

Press Enter, speak during the four-second recording, and read the `text` field in
the terminal. Enter `q` at the prompt to quit, or press Ctrl+C. For a single capture:

```bash
python3 voice_to_text.py --once
```

Microphone selection is fully automatic. The standalone
selector needs only Python and `alsa-utils`, not Whisper weights. It prefers
recognizable USB/webcam audio inputs, then tries the default/audio-server routes
and other capture devices. Each candidate must produce a valid one-second mono
16-bit/16kHz recording. Busy or incompatible inputs are skipped. Tests use a
bounded timeout, save only temporary audio, and do not alter system audio settings.

In standalone speech mode, selection happens before the first recording prompt.
In the camera view, it happens in the background on the first **P** request; wait
until the panel says `Recording` before speaking. The chosen device is reused.
If camera-view capture fails after disconnection, the next request selects again.

This verifies recording capability, not which microphone you prefer or whether
it is muted. Silence is allowed during the probe so a quiet room does not cause
every microphone to be rejected. No manual microphone/device argument is needed.
For a longer recording interval:

```bash
python3 voice_to_text.py --record-seconds 6
```

A USB webcam may not have an audio input. Check that a microphone is connected
and its capture level is set.
Near-silent recordings return an empty transcript. Audio and transcript files are
temporary and removed after each request; nothing is automatically archived.

An existing audio file can be transcribed without a microphone:

```bash
python3 voice_to_text.py --wav recording.wav
```

Files must be mono, 16-bit PCM WAV at 16 kHz and no longer than 120 seconds. If
FFmpeg is already installed, convert other recordings with:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le recording.wav
```

## Voice alongside the live object-depth view

Use the Python environment that already runs YOLO and depth:

```bash
python3 live_yolo_object_depth.py --voice
```

Click/focus the camera window, press **P**, and speak when the bottom panel shows
`Recording`. The four-second recording and Whisper transcription run in a
background worker; duplicate requests are rejected while one is active. The view
shows the recognized text and highlights a selected track in green.

Supported examples:

- `select the bottle`
- `find the cup`
- `pick up the bottle` (target selection only)
- `cancel` or `clear selection`

The object must be a supported YOLO class and uniquely visible with a tracking ID.
If multiple matching objects are visible, the request is reported as ambiguous.
An invisible target is not automatically replaced with another object. Speech
older than 15 seconds from the end of capture cannot update target selection.
Arbitrary phrases are still transcribed, but only the small explicit command
grammar affects selection. No voice command sends servo motion or performs a
physical pickup. `stop` clears the visual selection; it is not a motor emergency
stop. The hardware supervisor/confirmation path is not integrated here.

Press **Q**, **Esc**, or **Ctrl+C** to quit; an active speech process is cancelled.
For SSH without a display, use standalone `voice_to_text.py`. Other camera scripts
retain their existing behavior; their reference/stereo features are unchanged.

## Upgrade the model or use an existing runtime

```bash
bash tools/setup_whisper.sh base.en
python3 voice_to_text.py --whisper-model models/whisper/ggml-base.en.bin
python3 live_yolo_object_depth.py --voice --whisper-model models/whisper/ggml-base.en.bin
```

`small.en` is also accepted by the setup script. Changing model weights does not
change microphone capture or the command interface. Larger models use more
resources. Each utterance launches the CLI and reloads its model; measure response
time with vision running before increasing model size. A persistent Whisper
server would be a later performance improvement.

Pass `--whisper-bin /path/to/whisper-cli` and `--whisper-model /path/to/model.bin`
to use an existing compatible runtime instead of building another copy. The
wrapper explicitly requests CPU inference. `--speech-timeout` bounds transcription
waiting and `--whisper-threads` limits CPU workers.

## Verification

Automated tests use deterministic audio/recognizer fixtures for format checks,
silence, errors, bounded background work, stale requests, command parsing, and
ambiguous/lost targets. The wrapper was also checked with actual `tiny.en` weights
and the upstream `jfk.wav` sample on a Windows development host. That validates
local file transcription, not Jetson performance, ALSA capture, or live accuracy.
Test your microphone, command accuracy, memory, and camera responsiveness on the
actual Jetson. The speech implementation does not calibrate monocular depth.
