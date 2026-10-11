#!/usr/bin/env python3
"""Local AI pool coach: gentle spoken shot advice from the existing planner.

The billiards planner (billiards.py) remains the only source of geometry. A
local Hugging Face language model only rephrases the planner's numbers and maps
casual questions onto requests the existing parser accepts. Every model reply is
checked against the planner's numbers; a fixed template is used otherwise.
Speech uses a local Piper voice (or espeak) through the default audio output,
e.g. a paired JBL Bluetooth speaker. See COACH.md.
"""

import argparse
import json
import os
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

from billiards import BALLS, GROUPS, _rail_reference


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PIPER_BIN = ROOT / ".local/coach/piper/piper"
DEFAULT_PIPER_VOICE = ROOT / ".local/coach/voices/en_US-lessac-medium.onnx"
# Holes are numbered clockwise from top left in the calibrated table orientation.
HOLES = {"TL": 1, "TM": 2, "TR": 3, "BR": 4, "BM": 5, "BL": 6}
HOLE_POINTS = {"TL": (0, 0), "TM": (0.5, 0), "TR": (1, 0), "BR": (1, 1), "BM": (0.5, 1), "BL": (0, 1)}
INTENTS = ("best", "ball", "group", "why", "next", "unknown")
HELP_LINE = "I can help you pick a shot. Try asking, what's my best shot?"


def _number_text(value):
    return f"{value:g}"


def _half(value):
    return round(value * 2) / 2


def fullness(cut_angle_deg):
    """Plain-language hit description; ghost-ball overlap is 1 - sin(cut)."""
    if cut_angle_deg < 7:
        return "straight-in shot"
    if cut_angle_deg < 22:
        return "three-quarter ball hit"
    if cut_angle_deg < 39:
        return "half-ball hit"
    if cut_angle_deg < 60:
        return "quarter-ball hit"
    return "thin cut"


def shot_facts(shot, table):
    """Speakable facts derived only from one planner shot and the table size."""
    width, height = table["width"], table["height"]
    cue = (shot["cue_path"][0]["x"], shot["cue_path"][0]["y"])
    ghost = (shot["ghost"]["x"], shot["ghost"]["y"])
    behind = (2 * cue[0] - ghost[0], 2 * cue[1] - ghost[1])
    stance = _rail_reference(cue, behind, width, height)
    aim = shot["rail"]
    pocket_id = shot["pocket_id"]
    kind = "side" if pocket_id in ("TM", "BM") else "corner"
    return {
        "ball": shot["ball_id"],
        "ball_name": BALLS[shot["ball_id"]]["name"],
        "hole": HOLES[pocket_id],
        "pocket": f"{shot['pocket_name']} {kind} pocket",
        "stance_rail": stance["side"], "stance_diamond": _half(stance["diamond"]),
        "aim_rail": aim["side"], "aim_diamond": _half(aim["diamond"]),
        "hit": fullness(shot["cut_angle_deg"]),
    }


def template_line(facts):
    return (f"Nice and easy. Line up from diamond {_number_text(facts['stance_diamond'])} "
            f"on the {facts['stance_rail']} rail, aim toward diamond "
            f"{_number_text(facts['aim_diamond'])} on the {facts['aim_rail']} rail, and roll "
            f"the {facts['ball']} into hole {facts['hole']}, the {facts['pocket']}. "
            f"It's a {facts['hit']}, so take your time.")


def explain_facts(shot, alternatives):
    clearance = None
    match = next((r for r in shot["reasons"] if r.startswith("Minimum extra clearance")), None)
    if match:
        clearance = round(float(re.findall(r"[\d.]+", match)[0]))
    return {"ball": shot["ball_id"], "cut": round(shot["cut_angle_deg"]),
            "cue_cm": round(shot["distances"]["cue"]), "object_cm": round(shot["distances"]["object"]),
            "clearance_cm": clearance, "alternatives": max(0, alternatives - 1),
            "hit": fullness(shot["cut_angle_deg"])}


def explain_template(facts):
    line = (f"I like the {facts['ball']} because it's a {facts['hit']} of about {facts['cut']} degrees, "
            f"the cue ball only travels {facts['cue_cm']} centimeters, and the path is clear.")
    if facts["alternatives"]:
        line += " Ask for another option if you'd prefer something else."
    return line


def _numbers_in(text):
    return re.findall(r"\d+(?:\.\d+)?", text)


def check_reply(reply, allowed, required):
    """Accept a model reply only if every number in it came from the planner."""
    if not reply or len(reply) > 400:
        return False
    found = [_number_text(float(n)) for n in _numbers_in(reply)]
    return set(found) <= allowed and required <= set(found)


def canonical_question(intent, ball=None, group=None):
    """Map an interpreted intent onto a phrase the existing parse_question accepts."""
    if intent == "ball" and isinstance(ball, int) and 1 <= ball <= 15:
        return f"aim for ball {ball}"
    if group in GROUPS:
        return f"I am {group}, what is my best shot" if group != "open" else "open table, what is my best shot"
    if intent in ("best", "group", "ball"):
        return "what is my best shot"
    return None


def parse_intent(reply):
    """Extract {"intent","ball","group"} from model text; anything malformed is unknown."""
    unknown = {"intent": "unknown", "ball": None, "group": None}
    match = re.search(r"\{.*?\}", reply or "", re.S)
    if not match:
        return unknown
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return unknown
    if not isinstance(data, dict) or data.get("intent") not in INTENTS:
        return unknown
    ball = data.get("ball")
    ball = ball if isinstance(ball, int) and not isinstance(ball, bool) and 1 <= ball <= 15 else None
    group = data.get("group") if data.get("group") in GROUPS else None
    return {"intent": data["intent"], "ball": ball, "group": group}


def quick_intent(question):
    """Cheap rules for follow-ups that need no model."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    if "why" in words:
        return "why"
    if words & {"next", "another", "else", "alternative", "different", "other"}:
        return "next"
    return None


class CoachLLM:
    """Small local instruct model used only for wording and intent mapping."""

    def __init__(self, model_id=DEFAULT_MODEL, device="0", timeout=8.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        use_cuda = device != "cpu" and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{device}" if use_cuda and device.isdecimal() else
                                   ("cuda" if use_cuda else "cpu"))
        self.model_id, self.timeout = model_id, timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if use_cuda else torch.float32).to(self.device).eval()
        self._lock = threading.Lock()
        self.last_latency_ms = None
        print(f"Coach model: {model_id}; device {self.device}", flush=True)
        self.generate([{"role": "user", "content": "Say ready."}], 4)  # warm-up

    def generate(self, messages, max_new_tokens=60):
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with self._lock, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         max_time=self.timeout,
                                         pad_token_id=self.tokenizer.eos_token_id)
        self.last_latency_ms = (time.monotonic() - started) * 1000
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def phrase(self, facts):
        stance, aim = _number_text(facts["stance_diamond"]), _number_text(facts["aim_diamond"])
        messages = [
            {"role": "system", "content":
                "You are a calm, gentle pool coach speaking out loud to a player. Rewrite the shot "
                "facts as one or two short, encouraging spoken sentences. Always say the ball number. "
                "Write every number as digits exactly as given and use no other numbers. No lists or emojis."},
            {"role": "user", "content":
                "Ball: the 9 yellow stripe. Pocket: hole 4, the bottom right corner pocket. Line up from: diamond 1.5 "
                "on the top rail. Aim toward: diamond 6 on the bottom rail. Hit: half-ball hit."},
            {"role": "assistant", "content":
                "Take a breath and line up from diamond 1.5 on the top rail toward diamond 6 on the "
                "bottom rail. A smooth half-ball hit sends the 9 into hole 4, the bottom right corner."},
            {"role": "user", "content":
                f"Ball: the {facts['ball_name']}. Pocket: hole {facts['hole']}, the {facts['pocket']}. "
                f"Line up from: diamond {stance} on the {facts['stance_rail']} rail. "
                f"Aim toward: diamond {aim} on the {facts['aim_rail']} rail. Hit: {facts['hit']}."},
        ]
        reply = self.generate(messages, 70)
        allowed = {str(facts["ball"]), str(facts["hole"]), stance, aim}
        return reply if check_reply(reply, allowed, allowed) else None

    def explain(self, facts):
        messages = [
            {"role": "system", "content":
                "You are a calm, gentle pool coach. In one or two short spoken sentences, explain why "
                "this shot was chosen. Always name the ball number. Write numbers as digits exactly "
                "as given and no other numbers."},
            {"role": "user", "content": json.dumps({"ball": 12, "cut": 18, "cue_cm": 70, "object_cm": 45,
                                                    "clearance_cm": 9, "hit": "three-quarter ball hit"})},
            {"role": "assistant", "content":
                "The 12 is your friend here: an easy three-quarter ball hit of about 18 degrees, "
                "and the cue ball only rolls 70 centimeters with nothing in the way."},
            {"role": "user", "content": json.dumps({k: v for k, v in facts.items() if k != "alternatives"})},
        ]
        reply = self.generate(messages, 70)
        allowed = {str(v) for k, v in facts.items() if isinstance(v, int) and k != "alternatives"}
        return reply if check_reply(reply, allowed, {str(facts["ball"])}) else None

    def interpret(self, question):
        messages = [
            {"role": "system", "content":
                'Convert a pool player\'s request into JSON with keys "intent", "ball", "group". '
                'intent: "best" (best or easiest shot), "ball" (a specific ball number), "group" '
                '(they say they have solids or stripes), "why" (why that shot), "next" (another '
                'option), or "unknown" (not about choosing a shot). ball: integer 1-15 or null. '
                'group: "solids", "stripes", "open" or null. Reply with JSON only.'},
            {"role": "user", "content": "which one's easiest, I've got the big ones"},
            {"role": "assistant", "content": '{"intent": "group", "ball": null, "group": "stripes"}'},
            {"role": "user", "content": "can I make the green one, number six"},
            {"role": "assistant", "content": '{"intent": "ball", "ball": 6, "group": null}'},
            {"role": "user", "content": question[:300]},
        ]
        return parse_intent(self.generate(messages, 40))


class Voice:
    """Local text-to-speech through the default audio output (the paired JBL)."""

    def __init__(self, engine="auto", piper_bin=DEFAULT_PIPER_BIN, piper_voice=DEFAULT_PIPER_VOICE,
                 lead_in_s=0.4):
        piper_bin = Path(piper_bin) if Path(piper_bin).is_file() else (
            Path(shutil.which("piper")) if shutil.which("piper") else None)
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        self.engine = None
        if engine in ("auto", "piper") and piper_bin and Path(piper_voice).is_file():
            self.engine, self.piper = "piper", (piper_bin, Path(piper_voice))
        elif engine in ("auto", "espeak") and espeak:
            self.engine, self.espeak = "espeak", espeak
        self.player = shutil.which("paplay") or shutil.which("aplay")
        self.lead_in_s = lead_in_s
        self.process, self._file = None, None
        if self.engine is None:
            print(f"Voice unavailable for engine '{engine}': run tools/setup_coach.sh or install "
                  "espeak-ng. Guidance still appears on screen.", flush=True)
        elif self.player is None and sys.platform != "win32":
            print("No audio player found (paplay/aplay); install pulseaudio-utils or alsa-utils.", flush=True)

    def _synthesize(self, text, path):
        if self.engine == "piper":
            binary, voice = self.piper
            subprocess.run([str(binary), "--model", str(voice), "--output_file", str(path)],
                           input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=30)
        else:
            subprocess.run([self.espeak, "-s", "140", "-w", str(path), "--", text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
        # Prepend silence so a Bluetooth speaker waking from idle keeps the first word.
        with wave.open(str(path), "rb") as source:
            params, frames = source.getparams(), source.readframes(source.getnframes())
        silence = b"\0" * (int(params.framerate * self.lead_in_s) * params.sampwidth * params.nchannels)
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(silence + frames)

    def say(self, text, wait=False):
        self.stop()
        if self.engine is None or not text:
            return
        handle, name = tempfile.mkstemp(prefix="coach-", suffix=".wav")
        os.close(handle)
        self._file = Path(name)
        try:
            self._synthesize(text[:800], self._file)
        except (OSError, subprocess.SubprocessError, wave.Error) as exc:
            print(f"Speech synthesis failed: {exc}", flush=True)
            return
        if sys.platform == "win32" and self.player is None:
            import winsound
            winsound.PlaySound(str(self._file), winsound.SND_FILENAME
                               | (0 if wait else winsound.SND_ASYNC))
            return
        if self.player:
            self.process = subprocess.Popen([self.player, str(self._file)], stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if wait:
                self.process.wait()

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
        elif sys.platform == "win32" and self._file is not None:
            import winsound
            winsound.PlaySound(None, 0)
        if self._file is not None:
            try:
                self._file.unlink()
            except OSError:
                pass
            self._file = None


class CoachVoiceWorker:
    """One background job at a time: wording (possibly slow) then speech.

    stop() bumps a generation counter, so a line computed for a layout that has
    since changed is never spoken.
    """

    def __init__(self, voice, llm=None):
        self.voice, self.llm = voice, llm
        self.generation = 0
        self.last_line = ""
        self._jobs = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, compose):
        """compose(llm) returns the text to speak; it runs off the UI thread."""
        with self._lock:
            self.generation += 1
            generation = self.generation
            self.last_line = ""
            while not self._jobs.empty():
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
            self._jobs.put_nowait((generation, compose))

    def stop(self):
        with self._lock:
            self.generation += 1
            self.last_line = ""
        self.voice.stop()

    def _current(self, generation):
        with self._lock:
            return generation == self.generation

    def _run(self):
        while True:
            generation, compose = self._jobs.get()
            if not self._current(generation):
                continue
            try:
                text = compose(self.llm)
            except Exception as exc:  # A coach failure must never stop the camera app.
                print(f"Coach wording failed: {exc}", flush=True)
                continue
            if text and self._current(generation):
                self.last_line = text
                print(f"Coach: {text}", flush=True)
                self.voice.say(text)


def coached_line(llm, facts):
    reply = None
    if llm is not None:
        try:
            reply = llm.phrase(facts)
        except Exception as exc:
            print(f"Coach model failed; using template: {exc}", flush=True)
    return reply or template_line(facts)


def explained_line(llm, facts):
    reply = None
    if llm is not None:
        try:
            reply = llm.explain(facts)
        except Exception as exc:
            print(f"Coach model failed; using template: {exc}", flush=True)
    return reply or explain_template(facts)


def add_coach_arguments(parser):
    parser.add_argument("--coach-model", default=DEFAULT_MODEL,
                        help="Hugging Face instruct model; 'none' uses templates only")
    parser.add_argument("--coach-device", default="0", help="CUDA index or cpu for the coach model")
    parser.add_argument("--coach-timeout", type=float, default=8.0)
    parser.add_argument("--tts", choices=("auto", "piper", "espeak"), default="auto")
    parser.add_argument("--piper-bin", type=Path, default=DEFAULT_PIPER_BIN)
    parser.add_argument("--piper-voice", type=Path, default=DEFAULT_PIPER_VOICE)


def load_llm(args):
    if args.coach_model.lower() == "none":
        print("Coach model disabled; using fixed gentle templates.", flush=True)
        return None
    try:
        return CoachLLM(args.coach_model, args.coach_device, args.coach_timeout)
    except Exception as exc:
        print(f"Coach model unavailable ({exc}); using fixed gentle templates.", flush=True)
        return None


def main():
    from billiards import default_state, plan_shots
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--say", help="Speak this text through the default audio output and exit")
    parser.add_argument("--demo", action="store_true", help="Coach the built-in sample layout (no cameras)")
    parser.add_argument("--group", choices=GROUPS, default="solids")
    parser.add_argument("--question", default="What is my best shot?")
    parser.add_argument("--no-audio", action="store_true")
    add_coach_arguments(parser)
    args = parser.parse_args()
    voice = None if args.no_audio else Voice(args.tts, args.piper_bin, args.piper_voice)
    if args.say:
        if voice:
            voice.say(args.say, wait=True)
        return
    if not args.demo:
        parser.error("Use --say TEXT or --demo.")
    state = default_state()
    state["group"] = args.group
    llm = load_llm(args)
    question = args.question
    try:
        plan_shots(state, question)
    except ValueError:
        parsed = {"intent": quick_intent(question), "ball": None, "group": None}
        if parsed["intent"] is None:
            parsed = llm.interpret(question) if llm else {"intent": "unknown", "ball": None, "group": None}
        question = canonical_question(parsed["intent"], parsed["ball"], parsed["group"]) or "what is my best shot"
        print(f"Interpreted {parsed} -> {question!r}", flush=True)
    plan = plan_shots(state, question)
    if not plan["shots"]:
        print(plan["message"])
        return
    facts = shot_facts(plan["shots"][0], state["table"])
    print("Facts:", json.dumps(facts))
    print("Template:", template_line(facts))
    line = coached_line(llm, facts)
    if llm:
        print(f"Coach ({llm.last_latency_ms:.0f} ms):", line)
    print("Why:", explained_line(llm, explain_facts(plan["shots"][0], len(plan["shots"]))))
    if voice:
        voice.say(line, wait=True)


if __name__ == "__main__":
    main()
