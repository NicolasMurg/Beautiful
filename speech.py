"""Local Whisper speech recognition and bounded push-to-talk support.

Uses whisper.cpp on the CPU and ALSA arecord on Linux. Python needs only its
standard library. Audio/transcripts are temporary; nothing is sent to a service.
"""

import array
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave


ROOT = Path(__file__).resolve().parent
DEFAULT_BINARY = ROOT / ".local/whisper.cpp/build/bin/whisper-cli"
DEFAULT_MODEL = ROOT / "models/whisper/ggml-tiny.en.bin"


def add_speech_arguments(parser):
    parser.add_argument("--whisper-bin", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_MODEL,
                        help="Local ggml model; switch to ggml-base.en.bin to upgrade")
    parser.add_argument("--record-seconds", type=int, default=4,
                        help="Seconds captured per push-to-talk request (1-20)")
    parser.add_argument("--whisper-threads", type=int, default=2)
    parser.add_argument("--speech-timeout", type=float, default=90.0)


def checked_run(command, timeout, cancel=None):
    """Run a local process with cancellation, bounded waiting, and no shell."""
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Speech request cancelled")
    deadline = time.monotonic() + timeout
    with subprocess.Popen([str(value) for value in command], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace") as process:
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Speech request cancelled")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Speech process timed out after {timeout:g} seconds")
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                detail = (stderr or stdout).strip()[-2000:]
                raise RuntimeError(f"{Path(command[0]).name} failed: {detail}")
            return stdout
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
            raise


def wav_rms(path):
    """Check the required PCM format and return normalized RMS, rejecting empty audio."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("Audio must be mono, 16-bit PCM WAV at 16000 Hz")
        if audio.getnframes() > 16000 * 120:
            raise ValueError("Audio is limited to 120 seconds per request")
        samples = array.array("h", audio.readframes(audio.getnframes()))
    if not samples:
        raise ValueError("Audio file contains no samples")
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768.0


class WhisperCpp:
    def __init__(self, binary=DEFAULT_BINARY, model=DEFAULT_MODEL, threads=2, timeout=90):
        binary = str(binary)
        resolved = str(Path(binary).resolve()) if Path(binary).is_file() else shutil.which(binary)
        if resolved is None:
            raise FileNotFoundError("whisper-cli is missing; run bash tools/setup_whisper.sh first")
        self.binary = resolved
        self.model = Path(model).resolve()
        if not self.model.is_file():
            raise FileNotFoundError(f"Whisper model is missing: {self.model}")
        if threads < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Whisper threads and timeout must be positive")
        self.threads, self.timeout = threads, timeout

    def transcribe(self, wav_path, cancel=None):
        wav_path = Path(wav_path).resolve()
        # Reject near-silence before Whisper can invent a transcription for it.
        if wav_rms(wav_path) < 0.001:
            return ""
        with tempfile.TemporaryDirectory(prefix="whisper-text-") as temporary:
            output = Path(temporary) / "transcript"
            checked_run([self.binary, "-m", self.model, "-f", wav_path,
                         "-l", "en", "-t", str(self.threads), "-ng", "-nt",
                         "-otxt", "-of", output], self.timeout, cancel)
            transcript = output.with_suffix(".txt")
            if not transcript.is_file():
                raise RuntimeError("Whisper did not produce a transcript file")
            return " ".join(transcript.read_text(encoding="utf-8-sig").split())


def record_audio(path, seconds=4, device="default", cancel=None, timeout=None):
    if not 1 <= seconds <= 20:
        raise ValueError("Recording duration must be between 1 and 20 seconds")
    recorder = shutil.which("arecord")
    if recorder is None:
        raise FileNotFoundError("arecord is missing; install alsa-utils on the Jetson")
    checked_run([recorder, "--quiet", "--nonblock", "-D", device, "-f", "S16_LE", "-r", "16000",
                 "-c", "1", "-d", str(seconds), "-t", "wav", str(path)],
                seconds + 10 if timeout is None else timeout, cancel)


def resolve_microphone(cancel=None):
    """Automatically select an input that passes a recording test."""
    from microphone import choose_microphone

    def probe(path, candidate, timeout, cancellation):
        record_audio(path, 1, candidate, cancellation, timeout=timeout)

    print("Selecting a microphone; wait before speaking...", flush=True)
    return choose_microphone(probe, cancel, report=lambda text: print(text, flush=True)).device


@dataclass(frozen=True)
class SpeechResult:
    text: str
    recorded_at: float
    completed_at: float
    error: str = ""


class SpeechWorker:
    """At most one request and one pending result; no capture/inference on the UI thread."""
    def __init__(self, recognizer, seconds=4, device="auto"):
        if not 1 <= seconds <= 20:
            raise ValueError("Recording duration must be between 1 and 20 seconds")
        self.recognizer, self.seconds, self.device = recognizer, seconds, device
        self._auto_microphone = device == "auto"
        self._results = queue.Queue(maxsize=1)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._status = "Ready | P: record"

    @property
    def status(self):
        with self._lock:
            return self._status

    def _set_status(self, value):
        with self._lock:
            self._status = value

    def request(self):
        with self._lock:
            if (self._cancel.is_set() or not self._results.empty()
                    or self._thread is not None and self._thread.is_alive()):
                return False
            self._status = ("Choosing microphone; wait before speaking" if self.device == "auto"
                            else f"Recording for {self.seconds}s; speak now")
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()
            return True

    def _work(self):
        recorded_at = time.monotonic()
        try:
            if self.device == "auto":
                self.device = resolve_microphone(self._cancel)
            self._set_status(f"Recording for {self.seconds}s; speak now")
            with tempfile.TemporaryDirectory(prefix="whisper-audio-") as temporary:
                path = Path(temporary) / "microphone.wav"
                try:
                    record_audio(path, self.seconds, self.device, self._cancel)
                except Exception:
                    if self._auto_microphone:
                        self.device = "auto"  # Re-check after a disconnect on the next request.
                    raise
                recorded_at = time.monotonic()
                self._set_status("Transcribing locally on CPU")
                text = self.recognizer.transcribe(path, self._cancel)
            result = SpeechResult(text, recorded_at, time.monotonic())
        except Exception as exc:
            result = SpeechResult("", recorded_at, time.monotonic(), str(exc))
        if not self._cancel.is_set():
            self._results.put_nowait(result)
            self._set_status("Ready | P: record")

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def worker_from_args(args):
    if shutil.which("arecord") is None:
        raise FileNotFoundError("arecord is missing; install alsa-utils on the Jetson")
    return SpeechWorker(WhisperCpp(args.whisper_bin, args.whisper_model,
                                  args.whisper_threads, args.speech_timeout),
                        args.record_seconds)


def parse_object_request(text, supported_labels):
    """Accept a small command grammar; never infer a physical motion from speech."""
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if normalized in ("cancel", "cancel selection", "clear selection", "stop"):
        return "clear", None
    match = re.fullmatch(r"(?:pick up|select|find|grab)\s+(?:(?:the|a|an)\s+)?(.+)", normalized)
    if match is None:
        return "unrecognized", None
    requested = match.group(1)
    aliases = {"water bottle": "bottle", "coffee cup": "cup", "cellphone": "cell phone"}
    requested = aliases.get(requested, requested)
    labels = {label.lower(): label for label in supported_labels}
    if requested not in labels:
        return "unsupported", requested
    return "select", labels[requested]


class VoiceSelection:
    def __init__(self, supported_labels):
        self.supported_labels = supported_labels
        self.track_id = None
        self.object_name = None
        self.transcript = ""
        self.status = "Say: select the bottle"

    def apply(self, result, records, now=None):
        now = time.monotonic() if now is None else now
        if result.error:
            self.status = f"Speech error: {result.error}"
            return
        self.transcript = result.text
        if now - result.recorded_at > 15:
            self.status = "Speech result expired; repeat request"
            return
        action, target = parse_object_request(result.text, self.supported_labels)
        if action == "clear":
            self.track_id = self.object_name = None
            self.status = "Selection cleared"
        elif action == "select":
            self.track_id = self.object_name = None
            candidates = [r for r in records if r["object"] == target and r["track_id"] is not None]
            if len(candidates) == 1:
                self.track_id, self.object_name = candidates[0]["track_id"], target
                self.status = f"Selected {target} ID {self.track_id}; selection only"
            elif len(candidates) > 1:
                self.status = f"Multiple {target} objects visible; request is ambiguous"
            else:
                self.status = f"No tracked {target} visible; repeat when visible"
        elif action == "unsupported":
            self.status = f"YOLO does not support: {target}"
        else:
            self.status = "No supported request; say select the bottle"

    def draw(self, image, records, worker, cv2):
        visible = False
        for record in records:
            if (record["track_id"] == self.track_id and self.track_id is not None
                    and record["object"] == self.object_name):
                visible = True
                x1, y1, x2, y2 = (int(v) for v in record["bbox_xyxy"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        status = self.status
        if self.track_id is not None and not visible:
            status = f"Selected ID {self.track_id} not visible; no target switching"
        height, width = image.shape[:2]
        lines = [f"Voice: {worker.status}", f"Heard: {self.transcript}", status]
        top = max(0, height - 65)
        cv2.rectangle(image, (0, top), (width - 1, height - 1), (0, 0, 0), -1)
        for index, text in enumerate(lines):
            cv2.putText(image, text[:85], (8, top + 17 + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
