import array
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech


def make_wav(path, amplitude=1000, rate=16000, channels=1):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(array.array("h", [amplitude, -amplitude] * 800).tobytes())
    return path


def recognizer(tmp_path):
    model = tmp_path / "ggml-tiny.en.bin"
    model.write_bytes(b"test fixture; no actual inference")
    return speech.WhisperCpp(sys.executable, model)


def test_wav_rms_and_format_validation(tmp_path):
    path = make_wav(tmp_path / "audio.wav", amplitude=16384)
    assert speech.wav_rms(path) == 0.5
    with pytest.raises(ValueError, match="16000"):
        speech.wav_rms(make_wav(tmp_path / "wrong-rate.wav", rate=48000))
    with pytest.raises(ValueError, match="mono"):
        speech.wav_rms(make_wav(tmp_path / "stereo.wav", channels=2))


def test_silence_does_not_launch_whisper(tmp_path, monkeypatch):
    def must_not_run(*args, **kwargs):
        pytest.fail("Silent audio must not invoke a recognizer")
    monkeypatch.setattr(speech, "checked_run", must_not_run)
    assert recognizer(tmp_path).transcribe(make_wav(tmp_path / "silent.wav", 0)) == ""


def test_transcript_reads_output_file_not_whisper_logs(tmp_path, monkeypatch):
    captured = []
    def fake_run(command, timeout, cancel=None):
        captured.extend(command)
        output = Path(command[command.index("-of") + 1]).with_suffix(".txt")
        output.write_text("  Pick up\nthe bottle.\n", encoding="utf-8")
        return "system_info: CPU; timings, not speech"
    monkeypatch.setattr(speech, "checked_run", fake_run)
    path = make_wav(tmp_path / "audio with spaces.wav")
    assert recognizer(tmp_path).transcribe(path) == "Pick up the bottle."
    assert "-ng" in captured
    assert captured[captured.index("-f") + 1] == path.resolve()


def test_missing_transcript_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(speech, "checked_run", lambda *args, **kwargs: "")
    with pytest.raises(RuntimeError, match="produce a transcript"):
        recognizer(tmp_path).transcribe(make_wav(tmp_path / "audio.wav"))


def test_missing_model_and_invalid_resource_settings(tmp_path):
    with pytest.raises(FileNotFoundError, match="model is missing"):
        speech.WhisperCpp(sys.executable, tmp_path / "missing.bin")
    model = tmp_path / "model.bin"
    model.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        speech.WhisperCpp(sys.executable, model, threads=0)
    with pytest.raises(ValueError):
        speech.WhisperCpp(sys.executable, model, timeout=float("nan"))


def test_process_failure_timeout_and_cancellation():
    with pytest.raises(RuntimeError, match="failed"):
        speech.checked_run([sys.executable, "-c", "raise SystemExit(2)"], 5)
    with pytest.raises(RuntimeError, match="timed out"):
        speech.checked_run([sys.executable, "-c", "import time; time.sleep(10)"], 0.2)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        speech.checked_run([sys.executable, "-c", "pass"], 5, cancel)


@pytest.mark.parametrize("text,target", [("Pick up the bottle.", "bottle"),
                                       ("select a cup", "cup"),
                                       ("find the water bottle", "bottle")])
def test_positive_requests(text, target):
    assert speech.parse_object_request(text, ["bottle", "cup"]) == ("select", target)


@pytest.mark.parametrize("text", ["do not pick up the bottle", "I saw a bottle",
                                  "select the bottle and the cup", "please stop and grab a cup"])
def test_unrelated_negated_and_multiple_requests_do_not_select(text):
    action, _ = speech.parse_object_request(text, ["bottle", "cup"])
    assert action != "select"


def test_unsupported_class_is_explicit():
    assert speech.parse_object_request("select the screwdriver", ["bottle"]) == (
        "unsupported", "screwdriver")


def record(track_id, name="bottle"):
    return {"track_id": track_id, "object": name, "bbox_xyxy": [10, 10, 50, 80]}


def result(text="select the bottle", recorded_at=99, error=""):
    return speech.SpeechResult(text, recorded_at, 100, error)


def test_selection_requires_unique_current_track_and_clears_old_target():
    selection = speech.VoiceSelection(["bottle", "cup"])
    selection.apply(result(), [record(7)], now=100)
    assert selection.track_id == 7
    selection.apply(result(), [record(7), record(8)], now=100)
    assert selection.track_id is None
    assert "ambiguous" in selection.status
    selection.apply(result(), [], now=100)
    assert selection.track_id is None
    selection.apply(result(), [record(None)], now=100)
    assert selection.track_id is None


def test_stale_or_failed_speech_cannot_change_selection():
    selection = speech.VoiceSelection(["bottle", "cup"])
    selection.apply(result(), [record(7)], now=100)
    selection.apply(result("select the cup", recorded_at=1), [record(9, "cup")], now=100)
    assert selection.track_id == 7
    assert "expired" in selection.status
    selection.apply(result(error="microphone unavailable"), [], now=100)
    assert selection.track_id == 7
    assert "Speech error" in selection.status


def test_cancel_clears_selection_without_motion():
    selection = speech.VoiceSelection(["bottle"])
    selection.apply(result(), [record(7)], now=100)
    selection.apply(result("cancel"), [], now=100)
    assert selection.track_id is None


def test_lost_or_reclassified_target_is_not_replaced_in_view():
    selection = speech.VoiceSelection(["bottle", "cup"])
    selection.apply(result(), [record(7)], now=100)
    rectangles, labels = [], []
    cv2 = SimpleNamespace(
        rectangle=lambda *args: rectangles.append(args),
        putText=lambda *args: labels.append(args[1]),
        FONT_HERSHEY_SIMPLEX=0, LINE_AA=0)
    image = SimpleNamespace(shape=(480, 640, 3))
    selection.draw(image, [record(8), record(7, "cup")],
                   SimpleNamespace(status="Ready"), cv2)
    assert selection.track_id == 7
    assert all(args[3] != (0, 255, 0) for args in rectangles)
    assert any("not visible" in label for label in labels)


def test_worker_is_nonblocking_and_rejects_duplicate_requests(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def fake_record(path, seconds, device, cancel):
        entered.set()
        assert release.wait(2)
        make_wav(path)
    monkeypatch.setattr(speech, "record_audio", fake_record)
    engine = SimpleNamespace(transcribe=lambda path, cancel: "select the bottle")
    worker = speech.SpeechWorker(engine, device="test-fixture")
    try:
        assert worker.request()
        assert entered.wait(1)
        assert not worker.request()
        assert worker.poll() is None
        release.set()
        worker._thread.join(timeout=2)
        assert not worker._thread.is_alive()
        assert not worker.request()  # Don't overwrite an unconsumed result.
        assert worker.poll().text == "select the bottle"
        assert worker.poll() is None
    finally:
        release.set()
        worker.close()
    assert not worker.request()


def test_worker_reports_errors_and_close_cancels_active_recording(monkeypatch):
    def fail_record(*args):
        raise RuntimeError("microphone disconnected")
    monkeypatch.setattr(speech, "record_audio", fail_record)
    worker = speech.SpeechWorker(None, device="test-fixture")
    assert worker.request()
    worker._thread.join(timeout=2)
    assert "disconnected" in worker.poll().error
    worker.close()

    entered = threading.Event()
    def wait_for_cancel(path, seconds, device, cancel):
        entered.set()
        assert cancel.wait(2)
        raise RuntimeError("cancelled")
    monkeypatch.setattr(speech, "record_audio", wait_for_cancel)
    worker = speech.SpeechWorker(None, device="test-fixture")
    assert worker.request()
    assert entered.wait(1)
    worker.close()
    assert not worker._thread.is_alive()
    assert worker.poll() is None
