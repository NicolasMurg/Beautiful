#!/usr/bin/env python3
"""Local microphone -> Whisper -> sample-layout AI Coach -> targeted JBL test."""

from jetson_paths import PROJECT_ROOT, RUNTIME_ROOT
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

PROJECT = PROJECT_ROOT
ASSETS = RUNTIME_ROOT / 'voice-coach'
sys.path.insert(0, str(PROJECT))
from billiards import default_state, plan_shots
from coach import CoachLLM, Voice, canonical_question, shot_facts, template_line
from speech import WhisperCpp, record_audio, wav_rms
from jbl_speaker_test import ADDRESS, find_sink, run


def synthesize(voice, text, path):
    voice._synthesize(text, path)
    with wave.open(str(path), 'rb') as audio:
        if audio.getnframes() <= 0:
            raise RuntimeError('Piper produced empty audio')


def advice(llm, question):
    state = default_state()
    try:
        plan = plan_shots(state, question)
    except ValueError:
        intent = llm.interpret(question)
        canonical = canonical_question(intent['intent'], intent['ball'], intent['group'])
        if canonical is None:
            raise ValueError('Please ask for a shot, for example: What is my best shot?')
        print('Coach interpreted request:', canonical, flush=True)
        plan = plan_shots(state, canonical)
    if not plan['shots']:
        return plan['message'], 'planner', None
    facts = shot_facts(plan['shots'][0], state['table'])
    line = llm.phrase(facts)
    if line is None:
        print('Qwen wording failed the existing number check; using the planner template.', flush=True)
        return template_line(facts), 'validated-template', facts
    return line, 'qwen', facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Use generated speech, not a live question; no playback')
    parser.add_argument('--seconds', type=int, default=6)
    parser.add_argument('--mic', default='pipewire', help='ALSA capture route; default uses desktop microphone')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 20:
        parser.error('--seconds must be 1..20')
    print('SAMPLE TABLE LAYOUT: this test does not use live camera positions.', flush=True)
    sink = find_sink()
    if sink is None or 'Connected: yes' not in run('bluetoothctl', 'info', ADDRESS):
        raise RuntimeError('Connect JBL Flip Essential 2 before starting the coach test.')
    print('Output:', sink[1], '| microphone:', args.mic, flush=True)
    recognizer = WhisperCpp(ASSETS/'whisper.cpp/build/bin/whisper-cli', ASSETS/'ggml-tiny.en.bin')
    voice = Voice('piper', ASSETS/'piper/piper', ASSETS/'en_US-lessac-medium.onnx')
    if voice.engine != 'piper':
        raise RuntimeError('Piper runtime/voice missing')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('This test requires the verified CUDA environment for Qwen.')
    print('Loading existing CoachLLM with smaller Qwen2.5-0.5B-Instruct locally on CUDA...', flush=True)
    llm = CoachLLM('Qwen/Qwen2.5-0.5B-Instruct', timeout=15)
    with tempfile.TemporaryDirectory(prefix='voice-coach-test-') as tmp:
        wav_path = Path(tmp)/'question.wav'
        if args.self_test:
            source = Path(tmp)/'generated-question.wav'
            synthesize(voice, 'What is my best shot?', source)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-ar', '16000', '-ac', '1',
                            '-c:a', 'pcm_s16le', str(wav_path)], check=True, timeout=15)
            print('SELF TEST: transcribing Piper-generated speech, not microphone input.', flush=True)
        else:
            while True:
                answer = input('Enter to record a 6-second question, or Q to quit: ').strip().lower()
                if answer == 'q':
                    return
                print(f'RECORDING {args.seconds} seconds. Say: What is my best shot?', flush=True)
                record_audio(wav_path, args.seconds, args.mic)
                if wav_rms(wav_path) >= 0.001:
                    break
                print('Near-silent recording; speak near the Jieli webcam microphone and retry.', flush=True)
        rms = wav_rms(wav_path)
        started = time.monotonic()
        question = recognizer.transcribe(wav_path)
        stt_seconds = time.monotonic() - started
        print(f'Heard: {question} (Whisper {stt_seconds:.2f}s, RMS {rms:.4f})', flush=True)
        if not question:
            raise RuntimeError('Whisper returned no speech; repeat the test near the microphone.')
        if args.self_test and 'best shot' not in question.lower():
            raise RuntimeError('Generated-speech recognition did not recover the test question.')
        started = time.monotonic()
        line, wording, facts = advice(llm, question)
        coach_seconds = time.monotonic() - started
        print(f'Coach [{wording}, {coach_seconds:.2f}s]: {line}', flush=True)
        reply_path = Path(tmp)/'advice.wav'
        synthesize(voice, line, reply_path)
        if not args.self_test:
            sink = find_sink()
            if sink is None:
                raise RuntimeError('JBL output disappeared; refusing to play to another output.')
            run('wpctl', 'set-mute', str(sink[0]), '0')
            print('Playing AI Coach advice through JBL...', flush=True)
            run('pw-play', '--target', str(sink[2]), str(reply_path), timeout=60)
            if 'Connected: yes' not in run('bluetoothctl', 'info', ADDRESS):
                raise RuntimeError('JBL disconnected during coach playback')
        result = {'layout': 'built-in sample', 'input': 'generated speech' if args.self_test else 'microphone',
                  'question': question, 'rms': rms, 'stt_seconds': stt_seconds,
                  'coach_seconds': coach_seconds, 'wording': wording, 'facts': facts,
                  'advice': line, 'playback_completed': not args.self_test,
                  'audible_confirmation': 'pending' if not args.self_test else 'not played'}
        out = ASSETS/('self_test_result.json' if args.self_test else 'microphone_test_result.json')
        out.write_text(json.dumps(result, indent=2)+'\n')
        print('PASS:', out, flush=True)
        if not args.self_test:
            print('Microphone -> Whisper -> Coach -> Piper -> JBL completed. Confirm hearing the advice.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nStopped.', flush=True)
    except Exception as exc:
        print(f'Test failed: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
